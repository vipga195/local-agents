import unittest
import tempfile
import os
import shutil
import copy
import io
from contextlib import redirect_stderr
from pathlib import Path
import sqlite3
from memory import Memory
from brain import BrainError, tool_calls
from tools import ToolBox
from agent import run, build_messages, estimate_tokens, name_run

class FakeBrain:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((copy.deepcopy(messages), tools))
        item = self.replies.pop(0) if self.replies else BrainError('no more replies')
        if isinstance(item, BaseException):
            raise item
        return item

def tool_msg(name, args):
    return {'role':'assistant','content':'','tool_calls':[{'function':{'name':name,'arguments':args}}]}

def text_msg(s):
    return {'role':'assistant','content':s}

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'memory.db'
        self.root_dir = Path(self.temp_dir) / 'root'
        self.root_dir.mkdir()
        self.memory = Memory(self.db_path)
        self.brain = FakeBrain([])
        self.toolbox = ToolBox(self.memory, self.root_dir)

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir)

    def test_plain_reply_done(self):
        self.brain = FakeBrain([text_msg('hello')])
        result = run('say hi', self.brain, self.memory, self.toolbox)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'hello')
        run_id = result['run_id']
        row = self.memory.get_run(run_id)
        self.assertEqual(row['outcome'], 'done')
        self.assertEqual(len(row['steps']), 1)

    def test_list_dir_then_finish(self):
        (self.root_dir / 'a.txt').touch()
        self.brain = FakeBrain([
            tool_msg('list_dir', {'path': '.'}),
            tool_msg('finish', {'answer': 'done list'})
        ])
        result = run('list files', self.brain, self.memory, self.toolbox)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'done list')
        row = self.memory.get_run(result['run_id'])
        self.assertIn('a.txt', row['steps'][0]['results'][0])
        self.assertEqual(row['steps'][0]['calls'][0]['name'], 'list_dir')
        self.assertEqual(len(row['steps'][0]['results']), 1)

    def test_finish_ends_loop(self):
        self.brain = FakeBrain([
            tool_msg('finish', {'answer': 'x'}),
            text_msg('extra'),
            text_msg('more')
        ])
        result = run('finish now', self.brain, self.memory, self.toolbox)
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'x')

    def test_max_steps(self):
        self.brain = FakeBrain([tool_msg('list_dir', {'path': '.'})] * 10)
        result = run('list many', self.brain, self.memory, self.toolbox, max_steps=3)
        self.assertEqual(len(self.brain.calls), 3)
        self.assertEqual(result['outcome'], 'max_steps')
        self.assertEqual(result['answer'], '')
        run_id = result['run_id']
        row = self.memory.get_run(run_id)
        self.assertEqual(row['outcome'], 'max_steps')
        self.assertIsNotNone(row['finished'])

    def test_brain_error(self):
        self.brain = FakeBrain([BrainError('boom')])
        result = run('boom', self.brain, self.memory, self.toolbox)
        self.assertTrue(result['outcome'].startswith('error:'))
        self.assertEqual(result['answer'], '')
        run_id = result['run_id']
        row = self.memory.get_run(run_id)
        self.assertTrue(row['outcome'].startswith('error:'))

    def test_build_messages_profile_and_lesson(self):
        self.memory.set_profile('name', 'Huy')
        self.memory.add_lesson('ollama timeout on large prompt', 'use smaller num_ctx')
        msgs = build_messages('ollama timeout', self.memory)
        self.assertIn('\nUser profile:\n- name: Huy', msgs[0]['content'])
        self.assertIn('Possibly relevant lessons', msgs[0]['content'])
        self.assertIn('use smaller num_ctx', msgs[0]['content'])
        self.assertEqual(msgs[1], {'role': 'user', 'content': 'ollama timeout'})

    def test_add_lesson_tool_stores_proposed(self):
        self.brain = FakeBrain([
            tool_msg('add_lesson', {'situation': 's1 unique', 'rule': 'r1 unique'}),
            text_msg('ok')
        ])
        result = run('add lesson', self.brain, self.memory, self.toolbox)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'ok')
        row = self.memory.get_run(result['run_id'])
        hits = self.memory.search_lessons('unique', include_proposed=True)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['status'], 'proposed')
        self.assertEqual(hits[0]['rule'], 'r1 unique')
        self.assertIn('saved lesson #', row['steps'][0]['results'][0])

    def run_tokens(self, steps, warn_tokens):
        replies = [tool_msg('list_dir', {'path': '.'})] * (steps - 1) + [text_msg('ok')]
        self.brain = FakeBrain(replies)
        err = io.StringIO()
        with redirect_stderr(err):
            result = run('go', self.brain, self.memory, self.toolbox, warn_tokens=warn_tokens)
        return result, err.getvalue()

    def test_estimate_tokens_monotonic_and_counts_tools(self):
        small = estimate_tokens([{'role': 'user', 'content': 'a'}])
        big = estimate_tokens([{'role': 'user', 'content': 'a' * 300}])
        self.assertGreater(big, small)
        self.assertGreater(estimate_tokens([], [{'type': 'function'}]), estimate_tokens([], None))
        self.assertIsInstance(small, int)

    def test_warns_once_when_crossing_threshold(self):
        result, err = self.run_tokens(4, 1)
        self.assertEqual(err.count('warning:'), 1)
        self.assertIn('warning: session context ~', err)
        self.assertIn('(estimate, threshold 1)', err)
        self.assertEqual(result['outcome'], 'done')

    def test_no_warning_under_threshold_or_none(self):
        result, err = self.run_tokens(3, 10 ** 9)
        self.assertEqual(err, '')
        result, err = self.run_tokens(3, None)
        self.assertEqual(err, '')

    def test_tokens_is_peak_estimate(self):
        result, _ = self.run_tokens(3, None)
        calls = self.brain.calls
        want = max(estimate_tokens(m, t) for m, t in calls)
        self.assertEqual(result['tokens'], want)
        self.assertGreater(want, 0)

class TestFallback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir) / 'root'
        self.root_dir.mkdir()
        self.memory = Memory(Path(self.temp_dir) / 'memory.db')
        self.toolbox = ToolBox(self.memory, self.root_dir)

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir)

    def test_no_fallback_error_unchanged(self):
        main = FakeBrain([BrainError('x')])
        result = run('test', main, self.memory, self.toolbox)
        self.assertEqual(result['outcome'], 'error: x')
        self.assertEqual(result['answer'], '')
        self.assertEqual(len(main.calls), 1)

    def test_retry_main_once_no_route(self):
        main = FakeBrain([BrainError('a'), text_msg('ok')])
        fb = FakeBrain([])
        result = run('test', main, self.memory, self.toolbox, fallback=fb)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'ok')
        self.assertEqual(len(main.calls), 2)
        self.assertEqual(len(fb.calls), 0)
        self.assertEqual(self.memory.get_run(result['run_id'])['route'], '')

    def test_retry_does_not_use_extra_step(self):
        main = FakeBrain([BrainError('a'), text_msg('ok')])
        result = run('test', main, self.memory, self.toolbox, max_steps=1, fallback=FakeBrain([]))
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'ok')
        self.assertEqual(len(main.calls), 2)

    def test_fallback_after_two_main_failures(self):
        main = FakeBrain([BrainError('a'), BrainError('b')])
        fb = FakeBrain([text_msg('from fb')])
        result = run('test', main, self.memory, self.toolbox, fallback=fb)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'from fb')
        self.assertEqual(len(main.calls), 2)
        self.assertEqual(len(fb.calls), 1)
        self.assertEqual(self.memory.get_run(result['run_id'])['route'], 'fallback after 2 main failures: b')

    def test_main_and_fallback_fail(self):
        main = FakeBrain([BrainError('a'), BrainError('b')])
        fb = FakeBrain([BrainError('c')])
        result = run('test', main, self.memory, self.toolbox, fallback=fb)
        self.assertEqual(result['outcome'], 'error: c')
        self.assertEqual(result['answer'], '')
        self.assertEqual(len(main.calls), 2)
        self.assertEqual(len(fb.calls), 1)

    def test_counter_resets_after_success(self):
        main = FakeBrain([BrainError('a'), tool_msg('list_dir', {'path': '.'}), BrainError('b'), text_msg('ok')])
        fb = FakeBrain([])
        result = run('test', main, self.memory, self.toolbox, fallback=fb, max_steps=3)
        self.assertEqual(result['outcome'], 'done')
        self.assertEqual(result['answer'], 'ok')
        self.assertEqual(len(fb.calls), 0)
        self.assertEqual(self.memory.get_run(result['run_id'])['route'], '')

    def test_name_run_stores_cleaned_title(self):
        run_id = self.memory.start_run('goal')
        aux = FakeBrain([text_msg('  "Build a Pipeline"\nextra line')])
        name_run(self.memory, run_id, 'goal', aux)
        title = self.memory.get_run(run_id)['title']
        self.assertEqual(title, 'Build a Pipeline')
        self.assertIsNone(aux.calls[0][1])

    def test_name_run_truncates_to_80(self):
        run_id = self.memory.start_run('goal')
        aux = FakeBrain([text_msg('a' * 100)])
        name_run(self.memory, run_id, 'goal', aux)
        title = self.memory.get_run(run_id)['title']
        self.assertEqual(len(title), 80)

    def test_name_run_ignores_brain_error(self):
        run_id = self.memory.start_run('goal')
        aux = FakeBrain([BrainError('x')])
        name_run(self.memory, run_id, 'goal', aux)
        title = self.memory.get_run(run_id)['title']
        self.assertEqual(title, '')

    def test_name_run_empty_reply(self):
        run_id = self.memory.start_run('goal')
        aux = FakeBrain([text_msg('   ')])
        name_run(self.memory, run_id, 'goal', aux)
        title = self.memory.get_run(run_id)['title']
        self.assertEqual(title, '')

    def test_add_route_appends(self):
        run_id = self.memory.start_run('goal')
        self.memory.add_route(run_id, 'a')
        self.memory.add_route(run_id, 'b')
        route = self.memory.get_run(run_id)['route']
        self.assertEqual(route, 'a; b')

    def test_migration_adds_title_to_old_db(self):
        db = Path(self.temp_dir) / 'old.db'
        c = sqlite3.connect(str(db))
        c.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, goal TEXT NOT NULL, started REAL NOT NULL, finished REAL, outcome TEXT DEFAULT '', steps TEXT DEFAULT '[]', route TEXT DEFAULT '', user_verdict TEXT DEFAULT '')")
        c.execute("INSERT INTO runs (goal, started) VALUES ('old', 1.0)")
        c.commit()
        c.close()
        m = Memory(db)
        cols = [r['name'] for r in m.conn.execute('PRAGMA table_info(runs)')]
        self.assertIn('title', cols)
        self.assertEqual(m.get_run(1)['title'], '')
        m.set_title(1, 'T')
        self.assertEqual(m.get_run(1)['title'], 'T')
        m.close()

class TestToolBox(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir) / 'root'
        self.root_dir.mkdir()
        self.outside_dir = Path(self.temp_dir) / 'outside'
        self.outside_dir.mkdir()
        (self.outside_dir / 'secret.txt').write_text('SECRET')
        self.memory = Memory(Path(self.temp_dir) / 'memory.db')
        self.toolbox = ToolBox(self.memory, self.root_dir)

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir)

    def test_read_file_relative_path(self):
        result = self.toolbox.call('read_file', {'path': '../outside/secret.txt'})
        self.assertTrue(result.startswith('error:'))
        self.assertNotIn('SECRET', result)

    def test_read_file_absolute_path(self):
        result = self.toolbox.call('read_file', {'path': str(self.outside_dir / 'secret.txt')})
        self.assertTrue(result.startswith('error:'))

    def test_read_file_symlink(self):
        try:
            os.symlink(str(self.outside_dir / 'secret.txt'), str(self.root_dir / 'link.txt'))
            result = self.toolbox.call('read_file', {'path': 'link.txt'})
            self.assertTrue(result.startswith('error:'))
        except OSError:
            pass

    def test_unknown_tool(self):
        result = self.toolbox.call('nope', {})
        self.assertEqual(result, 'error: unknown tool nope')

    def test_missing_argument_path(self):
        result = self.toolbox.call('read_file', {})
        self.assertEqual(result, 'error: missing argument path')

    def test_large_file(self):
        (self.root_dir / 'large.txt').write_text('a' * 30000)
        result = self.toolbox.call('read_file', {'path': 'large.txt'})
        self.assertEqual(len(result), 20000)

    def test_exception_result(self):
        result = self.toolbox.call('list_dir', {'path': '../..'})
        self.assertTrue(isinstance(result, str))
        self.assertTrue(result.startswith('error:'))

    def test_graph_tools(self):
        c = self.toolbox.call
        self.assertTrue(c('add_node', {'type': 'script', 'name': 'Thạch Sanh', 'summary': 'cổ tích', 'data': {'p': 'x'}}).startswith('saved node script:Thạch Sanh (#'))
        c('add_node', {'type': 'character', 'name': 'Thạch Sanh nhân vật'})
        self.assertEqual(c('link_nodes', {'src_type': 'script', 'src_name': 'Thạch Sanh', 'rel': 'has_character', 'dst_type': 'character', 'dst_name': 'Thạch Sanh nhân vật'}),
                         'linked script:Thạch Sanh -has_character-> character:Thạch Sanh nhân vật')
        self.assertEqual(c('find_nodes', {'query': 'cổ tích', 'type': 'script'}), 'script:Thạch Sanh - cổ tích')
        self.assertEqual(c('find_nodes', {'query': 'zzz'}), 'no nodes found')
        self.assertEqual(c('get_node', {'type': 'script', 'name': 'Thạch Sanh'}),
                         'script:Thạch Sanh - cổ tích\ndata: {"p": "x"}\n-has_character-> character:Thạch Sanh nhân vật')
        self.assertEqual(c('get_node', {'type': 'script', 'name': 'nope'}), 'error: unknown node')
        self.assertTrue(c('link_nodes', {'src_type': 'a', 'src_name': 'b', 'rel': 'r', 'dst_type': 'c', 'dst_name': 'd'}).startswith('error:'))
        self.assertTrue(c('add_node', {'type': 's', 'name': 'n', 'data': 'bad'}).startswith('error:'))
        self.assertEqual(c('add_node', {'type': 'script'}), 'error: missing argument name')
        self.assertEqual(c('link_nodes', {'src_type': 'a'}), 'error: missing argument src_name')
        self.assertEqual(c('find_nodes', {}), 'error: missing argument query')
        self.assertEqual(c('get_node', {'type': 'a'}), 'error: missing argument name')

    def test_distilled_node_tools(self):
        c = self.toolbox.call
        nid = self.memory.add_node('script', 'Slim', 'sum', {'p': 'x'})
        path = os.path.join(self.root_dir, '..', 'slim-file.md')
        path = os.path.abspath(path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('# Slim\nfull content')
        try:
            self.memory.slim_nodes([(nid, path)])
            self.assertEqual(c('find_nodes', {'query': 'Slim'}), 'script:Slim - sum [distilled]')
            out = c('get_node', {'type': 'script', 'name': 'Slim'})
            self.assertEqual(out, f'script:Slim - sum\ndistilled: {path}\n# Slim\nfull content')
            self.assertTrue(c('add_node', {'type': 'script', 'name': 'Slim', 'summary': 'x'}).startswith('error: node is distilled'))
        finally:
            os.unlink(path)
        self.assertEqual(c('get_node', {'type': 'script', 'name': 'Slim'}), f'script:Slim - sum\ndistilled file missing: {path}')

if __name__ == '__main__':
    unittest.main()

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from lessons import main as lessons_main
from agent import run, build_messages, DRAWTHINGS_PROMPT
from mcp_client import McpError
from memory import Memory
from test_agent import FakeBrain, tool_msg, text_msg
from tools import ToolBox

TOOLS = [
    {'name': 'txt2img', 'description': 'gen', 'inputSchema': {'type': 'object', 'properties': {'prompt': {'type': 'string'}}, 'required': ['prompt']}},
    {'name': 'send_telegram_image', 'description': 'send', 'inputSchema': {'type': 'object', 'required': ['path']}},
    {'name': 'ask_choice', 'description': 'ask', 'inputSchema': {'type': 'object', 'required': ['question', 'options']}},
    {'name': 'img2img', 'description': 'nope', 'inputSchema': {'type': 'object'}},
]


class StubMcp:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []
        self.list_count = 0

    def list_tools(self):
        self.list_count += 1
        return TOOLS

    def call_tool(self, name, arguments, timeout):
        self.calls.append((name, arguments, timeout))
        r = self.results.get(name, 'ok')
        if isinstance(r, BaseException):
            raise r
        return r


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.memory = Memory(Path(self.dir) / 'm.db')

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.dir)

    def box(self, **results):
        self.mcp = StubMcp(results)
        tb = ToolBox(self.memory, self.dir, mcp=self.mcp)
        tb.begin_run(1, 'compare cats')
        return tb


class TestMcpToolBox(Base):
    def test_no_mcp_unchanged(self):
        tb = ToolBox(self.memory, self.dir)
        self.assertFalse(tb.has_mcp)
        self.assertEqual(len(tb.specs()), 10)
        self.assertEqual(tb.call('txt2img', {'prompt': 'x'}), 'error: unknown tool txt2img')

    def test_specs_schema_and_allowlist(self):
        tb = self.box()
        specs = tb.specs()
        names = [s['function']['name'] for s in specs]
        self.assertEqual(names[10:], ['txt2img', 'send_telegram_image', 'ask_choice'])
        self.assertNotIn('img2img', names)
        s = specs[10]
        self.assertEqual(s['type'], 'function')
        self.assertEqual(s['function']['parameters'], TOOLS[0]['inputSchema'])
        tb.specs()
        self.assertEqual(self.mcp.list_count, 1)

    def test_not_allowed_not_callable(self):
        tb = self.box()
        self.assertEqual(tb.call('img2img', {}), 'error: unknown tool img2img')
        self.assertEqual(self.mcp.calls, [])

    def test_required_arg(self):
        tb = self.box()
        self.assertEqual(tb.call('txt2img', {}), 'error: missing argument prompt')
        self.assertEqual(self.mcp.calls, [])

    def test_errors_become_strings(self):
        tb = self.box(txt2img=McpError('boom'))
        self.assertEqual(tb.call('txt2img', {'prompt': 'x'}), 'error: boom')
        self.assertEqual(tb.generated, [])

    def test_timeouts(self):
        tb = self.box()
        tb.call('txt2img', {'prompt': 'x'})
        tb.call('send_telegram_image', {'path': 'p'})
        tb.call('ask_choice', {'question': 'q', 'options': ['A']})
        tb.call('ask_choice', {'question': 'q', 'options': ['A'], 'timeout_s': 60})
        tb.call('ask_choice', {'question': 'q', 'options': ['A'], 'timeout_s': 'bad'})
        self.assertEqual([c[2] for c in self.mcp.calls], [3600, 300, 1920, 180, 1920])

    def test_ask_choice_records_one_proposed_lesson(self):
        tb = self.box(txt2img='img1.png', ask_choice='B')
        tb.call('txt2img', {'prompt': 'cat'})
        out = tb.call('ask_choice', {'question': 'Which?', 'options': ['A', 'B']})
        lessons = self.memory.list_lessons()
        self.assertEqual(len(lessons), 1)
        l = lessons[0]
        self.assertEqual(l['status'], 'proposed')
        self.assertEqual(l['situation'], 'compare cats')
        self.assertIn("User picked 'B' when asked: Which? (options: A, B)", l['rule'])
        self.assertIn('img1.png', l['evidence'])
        self.assertEqual(l['confidence'], 0.4)
        self.assertEqual(out, f"B\n(recorded as proposed lesson #{l['id']})")

    def test_expired_and_empty_record_nothing(self):
        for val in ('expired', ' Expired ', '', '  '):
            tb = self.box(ask_choice=val)
            self.assertEqual(tb.call('ask_choice', {'question': 'q', 'options': ['A']}), val)
        self.assertEqual(self.memory.list_lessons(), [])


class TestRunWithMcp(Base):
    def test_run_records_lesson(self):
        mcp = StubMcp({'txt2img': 'a.png', 'ask_choice': 'A'})
        tb = ToolBox(self.memory, self.dir, mcp=mcp)
        brain = FakeBrain([
            tool_msg('txt2img', {'prompt': 'cat'}),
            tool_msg('ask_choice', {'question': 'Pick', 'options': ['A', 'B']}),
            tool_msg('finish', {'answer': 'you chose A'}),
        ])
        res = run('compare cats', brain, self.memory, tb)
        self.assertEqual(res['outcome'], 'done')
        lessons = self.memory.list_lessons()
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]['status'], 'proposed')
        self.assertIn(DRAWTHINGS_PROMPT, brain.calls[0][0][0]['content'])

    def test_prompt_absent_without_mcp(self):
        brain = FakeBrain([text_msg('hi')])
        run('x', brain, self.memory, ToolBox(self.memory, self.dir))
        self.assertNotIn(DRAWTHINGS_PROMPT, brain.calls[0][0][0]['content'])
        self.assertNotIn('extra', build_messages('x', self.memory)[0]['content'])


class TestLessonsCli(Base):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = lessons_main(['--db', str(Path(self.dir) / 'm.db'), *argv])
        return code, out.getvalue(), err.getvalue()

    def test_list_approve_reject(self):
        a = self.memory.add_lesson('sit A', 'rule A', confidence=0.4)
        b = self.memory.add_lesson('sit B', 'rule B')
        code, out, _ = self.run_cli('list')
        self.assertEqual(code, 0)
        self.assertIn(f'#{a} [proposed] conf=0.40 | rule A', out)
        self.assertIn('    sit A', out)
        self.assertLess(out.index('rule B'), out.index('rule A'))
        self.assertEqual(self.run_cli('approve', str(a))[0], 0)
        self.assertEqual(self.run_cli('reject', str(b))[0], 0)
        _, out, _ = self.run_cli('list', '--status', 'accepted')
        self.assertIn('rule A', out)
        self.assertNotIn('rule B', out)
        self.assertIn('rule B', self.run_cli('list', '--status', 'rejected', '--limit', '5')[1])

    def test_unknown_id(self):
        code, out, err = self.run_cli('approve', '999')
        self.assertEqual(code, 1)
        self.assertTrue(err)
        self.assertEqual(out, '')


if __name__ == '__main__':
    unittest.main()

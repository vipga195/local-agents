import os
import sqlite3
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import distill
from memory import Memory


class TestDistill(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.db_path = os.path.join(self.dir.name, 'test.db')
        self.out = os.path.join(self.dir.name, 'out')
        self.memory = Memory(self.db_path)
        self.s = self.memory.add_node('script', 'Thạch Sanh', 'cổ tích', {'k': 1})
        self.c = self.memory.add_node('character', 'Lý Thông', 'phản diện')
        self.r = self.memory.add_node('character', 'Rejected', 'no')
        self.memory.link('script', 'Thạch Sanh', 'has_character', 'character', 'Lý Thông')
        self.memory.set_node_status(self.r, 'rejected')

    def tearDown(self):
        self.memory.close()
        self.dir.cleanup()

    def run_main(self, *args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = distill.main(['--db', self.db_path, *args])
        return code, out.getvalue(), err.getvalue()

    def read(self, *parts):
        with open(os.path.join(self.out, *parts), encoding='utf-8') as f:
            return f.read()

    def test_status_default_and_preserved(self):
        self.assertEqual(self.memory.get_node('script', 'Thạch Sanh')['status'], 'proposed')
        self.memory.set_node_status(self.s, 'accepted')
        self.memory.add_node('script', 'Thạch Sanh', 'v2')
        self.assertEqual(self.memory.get_node('script', 'Thạch Sanh')['status'], 'accepted')

    def test_migration_old_schema(self):
        path = os.path.join(self.dir.name, 'old.db')
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, name TEXT NOT NULL, summary TEXT DEFAULT '', data TEXT DEFAULT '{}', created REAL, updated REAL, UNIQUE(type, name))")
        conn.execute("INSERT INTO nodes (type, name) VALUES ('a', 'b')")
        conn.commit()
        conn.close()
        for _ in range(2):
            m = Memory(path)
            self.assertEqual(m.get_node('a', 'b')['status'], 'proposed')
            m.close()

    def test_export_accepted_only_by_default(self):
        self.memory.set_node_status(self.s, 'accepted')
        code, out, _ = self.run_main('export', '--out', self.out)
        self.assertEqual(code, 0)
        self.assertIn('exported 1', out)
        self.assertTrue(os.path.exists(os.path.join(self.out, 'script', 'thach-sanh.md')))
        self.assertFalse(os.path.exists(os.path.join(self.out, 'character', 'ly-thong.md')))

    def test_include_proposed_never_rejected(self):
        self.memory.set_node_status(self.s, 'accepted')
        self.run_main('export', '--out', self.out, '--include-proposed')
        self.assertTrue(os.path.exists(os.path.join(self.out, 'character', 'ly-thong.md')))
        self.assertFalse(os.path.exists(os.path.join(self.out, 'character', 'rejected.md')))

    def test_slug(self):
        self.assertEqual(distill.slugify('Thạch Sanh'), 'thach-sanh')
        self.assertEqual(distill.slugify('Đêm'), 'dem')
        self.assertEqual(distill.slugify('!!!'), '')

    def test_frontmatter_body_and_index(self):
        self.memory.set_node_status(self.s, 'accepted')
        self.run_main('export', '--out', self.out, '--include-proposed')
        text = self.read('script', 'thach-sanh.md')
        self.assertIn('has_character: ["Lý Thông"]', text)
        self.assertIn('name: "Thạch Sanh"', text)
        self.assertIn('# Thạch Sanh\n\ncổ tích', text)
        self.assertIn('## Data', text)
        index = self.read('index.md')
        self.assertIn('- [script:Thạch Sanh](script/thach-sanh.md) - cổ tích', index)
        self.assertNotIn('Rejected', index)

    def test_idempotent(self):
        self.memory.set_node_status(self.s, 'accepted')
        self.run_main('export', '--out', self.out)
        first = self.read('script', 'thach-sanh.md'), self.read('index.md')
        self.run_main('export', '--out', self.out)
        self.assertEqual(first, (self.read('script', 'thach-sanh.md'), self.read('index.md')))

    def test_slug_collision_and_empty(self):
        a = self.memory.add_node('x', 'A b')
        b = self.memory.add_node('x', 'a-b')
        c = self.memory.add_node('x', '!!!')
        for i in (a, b, c):
            self.memory.set_node_status(i, 'accepted')
        self.run_main('export', '--out', self.out)
        files = sorted(os.listdir(os.path.join(self.out, 'x')))
        self.assertEqual(files, sorted(['a-b.md', f'a-b-{b}.md', f'node-{c}.md']))

    def test_kb_refusal(self):
        target = os.path.join(self.dir.name, 'knowledge-base', 'sub')
        code, _, err = self.run_main('export', '--out', target)
        self.assertEqual(code, 2)
        self.assertIn('knowledge-base', err)
        self.assertFalse(os.path.exists(target))

    def test_list_approve_reject(self):
        self.assertEqual(self.run_main('approve', str(self.s))[0], 0)
        _, out, _ = self.run_main('list', '--status', 'accepted')
        self.assertEqual(out.strip(), '#1 [accepted] script:Thạch Sanh - cổ tích')
        self.assertEqual(self.run_main('reject', '999')[0], 1)

    def accept_all_but(self, *skip):
        for i in (self.s, self.c):
            if i not in skip:
                self.memory.set_node_status(i, 'accepted')

    def count(self, table):
        return self.memory.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]

    def test_export_without_slim_changes_nothing(self):
        self.accept_all_but()
        self.run_main('export', '--out', self.out)
        self.assertEqual(self.count('nodes'), 3)
        self.assertEqual(self.memory.get_node('script', 'Thạch Sanh')['distilled_path'], '')

    def test_slim_keeps_node_edges_summary_and_backs_up(self):
        self.accept_all_but()
        code, out, _ = self.run_main('export', '--out', self.out, '--slim')
        self.assertEqual(code, 0)
        self.assertIn('slimmed 2 nodes', out)
        self.assertEqual(self.count('nodes'), 3)
        self.assertEqual(self.count('edges'), 1)
        node = self.memory.get_node('script', 'Thạch Sanh')
        self.assertEqual(node['data'], {})
        self.assertEqual(node['summary'], 'cổ tích')
        self.assertEqual(node['distilled_path'], str(Path(self.out).resolve() / 'script' / 'thach-sanh.md'))
        self.assertEqual(len(node['out']), 1)
        self.assertEqual(len(self.memory.find_nodes('Thạch')), 1)
        self.assertTrue(os.path.getsize(self.db_path + '.bak') > 0)
        bak = Memory(self.db_path + '.bak')
        self.assertEqual(bak.get_node('script', 'Thạch Sanh')['data'], {'k': 1})
        bak.close()

    def test_slim_never_touches_proposed_or_rejected(self):
        self.memory.set_node_status(self.s, 'accepted')
        self.run_main('export', '--out', self.out, '--include-proposed', '--slim')
        paths = {n['name']: n['distilled_path'] != '' for n in self.memory.list_nodes()}
        self.assertEqual(paths, {'Thạch Sanh': True, 'Lý Thông': False, 'Rejected': False})

    def test_slim_aborts_when_file_missing(self):
        self.accept_all_but()
        real = distill.export_nodes

        def broken(*a, **k):
            exported = real(*a, **k)
            exported[0]['path'].unlink()
            return exported

        distill.export_nodes = broken
        try:
            code, _, err = self.run_main('export', '--out', self.out, '--slim')
        finally:
            distill.export_nodes = real
        self.assertEqual(code, 1)
        self.assertIn('nothing changed', err)
        self.assertEqual(self.memory.get_node('script', 'Thạch Sanh')['data'], {'k': 1})
        self.assertEqual(self.memory.get_node('script', 'Thạch Sanh')['distilled_path'], '')
        self.assertFalse(os.path.exists(self.db_path + '.bak'))

    def test_reexport_skips_stubs_and_is_idempotent(self):
        self.accept_all_but()
        self.run_main('export', '--out', self.out, '--slim')
        first = self.read('script', 'thach-sanh.md'), self.read('index.md')
        code, out, _ = self.run_main('export', '--out', self.out, '--slim')
        self.assertEqual(code, 0)
        self.assertIn('exported 0', out)
        self.assertIn('slimmed 0', out)
        self.assertEqual(first, (self.read('script', 'thach-sanh.md'), self.read('index.md')))
        self.assertIn('"k": 1', self.read('script', 'thach-sanh.md'))
        self.assertIn('- [script:Thạch Sanh](script/thach-sanh.md) - cổ tích', self.read('index.md'))

    def test_add_node_on_stub_raises(self):
        self.accept_all_but()
        self.run_main('export', '--out', self.out, '--slim')
        with self.assertRaises(ValueError):
            self.memory.add_node('script', 'Thạch Sanh', 'new', {'z': 1})
        self.assertEqual(self.memory.get_node('script', 'Thạch Sanh')['summary'], 'cổ tích')

    def test_migration_adds_distilled_path(self):
        path = os.path.join(self.dir.name, 'old2.db')
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, name TEXT NOT NULL, summary TEXT DEFAULT '', data TEXT DEFAULT '{}', status TEXT NOT NULL DEFAULT 'proposed', created REAL, updated REAL, UNIQUE(type, name))")
        conn.execute("INSERT INTO nodes (type, name) VALUES ('a', 'b')")
        conn.commit()
        conn.close()
        for _ in range(2):
            m = Memory(path)
            self.assertEqual(m.get_node('a', 'b')['distilled_path'], '')
            m.close()

    def test_distill_session_noop_without_accepted(self):
        self.assertIsNone(distill.distill_session(self.memory, self.out))
        self.assertFalse(os.path.exists(self.out))

    def test_distill_session_exports_and_slims(self):
        self.accept_all_but()
        line = distill.distill_session(self.memory, self.out)
        self.assertIn('slimmed 2', line)
        self.assertEqual(self.count('nodes'), 3)
        self.assertNotEqual(self.memory.get_node('character', 'Lý Thông')['distilled_path'], '')

    def test_distill_session_swallows_errors(self):
        self.accept_all_but()
        blocker = os.path.join(self.dir.name, 'file')
        open(blocker, 'w').close()
        err = StringIO()
        with redirect_stderr(err):
            self.assertIsNone(distill.distill_session(self.memory, os.path.join(blocker, 'sub')))
        self.assertIn('distill failed', err.getvalue())
        self.assertEqual(self.count('nodes'), 3)

    def test_distill_session_kb_refusal(self):
        self.accept_all_but()
        err = StringIO()
        with redirect_stderr(err):
            self.assertIsNone(distill.distill_session(self.memory, os.path.join(self.dir.name, 'knowledge-base')))
        self.assertIn('knowledge-base', err.getvalue())
        self.assertEqual(self.count('nodes'), 3)


    def add_adr(self, title='Dùng SQLite', **kw):
        from tools import ToolBox
        tb = ToolBox(self.memory, self.dir.name)
        args = dict(title=title, summary='chọn sqlite', context='bối cảnh', decision='dùng sqlite', tradeoff='đơn giản', tags='a, b')
        args.update(kw)
        res = tb.call('add_adr', args)
        return tb, res, self.memory.get_node('adr', title)

    def test_add_adr_tool(self):
        tb, res, node = self.add_adr()
        self.assertTrue(res.startswith('saved adr "Dùng SQLite" (#'))
        self.assertTrue(res.endswith(', proposed)'))
        self.assertEqual(node['status'], 'proposed')
        d = node['data']
        self.assertEqual((d['adr_type'], d['tags'], d['prs'], d['decision']), ('architecture', ['a', 'b'], [], 'dùng sqlite'))
        self.assertRegex(d['date'], r'^\d{4}-\d{2}-\d{2}$')
        self.assertEqual(tb.call('add_adr', {'title': 'x'}), 'error: missing argument summary')
        self.assertTrue(tb.call('add_adr', {'title': 'x', 'summary': 's', 'context': 'c', 'decision': 'd', 'type': 'bogus'}).startswith('error:'))

    def git(self, cwd, *args):
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=cwd, check=True, capture_output=True)

    def make_repo(self):
        repo = os.path.join(self.dir.name, 'repo')
        remote = os.path.join(self.dir.name, 'remote.git')
        os.makedirs(repo)
        self.git(self.dir.name, 'init', '--bare', remote)
        self.git(repo, 'init', '-b', 'main')
        self.git(repo, 'remote', 'add', 'origin', remote)
        return repo

    def commit(self, repo, rel, msg, push=True):
        Path(repo, rel).parent.mkdir(parents=True, exist_ok=True)
        Path(repo, rel).write_text(msg, encoding='utf-8')
        self.git(repo, 'add', '.')
        self.git(repo, 'commit', '-m', msg)
        if push:
            self.git(repo, 'push', '-u', 'origin', 'main')

    def accepted_adr(self, repo, paths='src'):
        _, _, node = self.add_adr(repo=repo, paths=paths)
        self.memory.set_node_status(node['id'], 'accepted')
        # commits made right after the node must count: move baseline back
        self.memory.conn.execute('UPDATE nodes SET created=created-3600 WHERE id=?', (node['id'],))
        self.memory.conn.commit()
        return node

    def adr_files(self):
        d = os.path.join(self.out, 'adr')
        return os.listdir(d) if os.path.isdir(d) else []

    def test_add_adr_stores_paths_and_repo(self):
        _, _, node = self.add_adr(paths='a.py, src/')
        self.assertEqual(node['data']['paths'], ['a.py', 'src/'])
        self.assertEqual(node['data']['repo'], str(Path(self.dir.name).resolve()))

    def test_unpushed_commit_exported_without_prs(self):
        repo = self.make_repo()
        self.commit(repo, 'init.txt', 'init')
        self.accepted_adr(repo)
        self.commit(repo, 'src/a.py', 'change (#3)', push=False)
        self.run_main('export', '--out', self.out, '--slim')
        self.assertIn('prs: []', self.read('adr', self.adr_files()[0]))

    def test_pushed_commit_exported_with_pr_and_slimmed(self):
        repo = self.make_repo()
        self.commit(repo, 'init.txt', 'init')
        node = self.accepted_adr(repo)
        self.commit(repo, 'src/a.py', 'change (#12)')
        self.run_main('export', '--out', self.out, '--slim')
        date = node['data']['date']
        text = self.read('adr', f'{date}-dung-sqlite.md')
        for want in ('project: local-agent', 'component: agent', 'type: architecture', 'status: accepted', f'date: {date}',
                     'summary: "chọn sqlite"', 'tags: [a, b]', 'related: []', 'prs: [12]', 'environments_affected: []',
                     '# Dùng SQLite', '## Bối cảnh', '## Quyết định', '## Lý do / Trade-off'):
            self.assertIn(want, text)
        n = self.memory.get_node('adr', 'Dùng SQLite')
        self.assertEqual(n['data'], {})
        self.assertTrue(n['distilled_path'].endswith(f'{date}-dung-sqlite.md'))
        self.assertIn('## Quyết định', Path(n['distilled_path']).read_text(encoding='utf-8'))
        _, out, _ = self.run_main('export', '--out', self.out, '--slim')
        self.assertIn('exported 0', out)
        self.assertIn('adr/', self.read('index.md'))

    def test_direct_push_without_pr_number(self):
        repo = self.make_repo()
        self.commit(repo, 'init.txt', 'init')
        self.accepted_adr(repo)
        self.commit(repo, 'src/a.py', 'direct push')
        self.run_main('export', '--out', self.out, '--adr-project', 'p', '--adr-component', 'c')
        text = self.read('adr', self.adr_files()[0])
        self.assertIn('prs: []', text)
        self.assertIn('project: p\ncomponent: c', text)

    def test_pushed_commit_not_touching_path(self):
        repo = self.make_repo()
        self.commit(repo, 'init.txt', 'init')
        self.accepted_adr(repo)
        self.commit(repo, 'other/b.py', 'unrelated (#4)')
        self.run_main('export', '--out', self.out)
        self.assertIn('prs: []', self.read('adr', self.adr_files()[0]))

    def test_proposed_adr_never_exported(self):
        repo = self.make_repo()
        self.commit(repo, 'init.txt', 'init')
        _, _, node = self.add_adr(repo=repo, paths='src')
        self.memory.conn.execute('UPDATE nodes SET created=created-3600 WHERE id=?', (node['id'],))
        self.memory.conn.commit()
        self.commit(repo, 'src/a.py', 'change (#5)')
        self.run_main('export', '--out', self.out, '--include-proposed')
        self.assertEqual(self.adr_files(), [])

    def test_non_git_dir_exported_without_prs(self):
        _, _, node = self.add_adr()
        self.memory.set_node_status(node['id'], 'accepted')
        self.run_main('export', '--out', self.out)
        self.assertIn('prs: []', self.read('adr', self.adr_files()[0]))

    def test_non_adr_unaffected(self):
        self.memory.set_node_status(self.s, 'accepted')
        self.run_main('export', '--out', self.out)
        self.assertTrue(os.path.exists(os.path.join(self.out, 'script', 'thach-sanh.md')))

    def test_hook_exports_accepted_adr_without_commit(self):
        repo = self.make_repo()
        self.commit(repo, 'init.txt', 'init')
        self.accepted_adr(repo)
        self.assertIn('slimmed 1', distill.distill_session(self.memory, self.out))


if __name__ == '__main__':
    unittest.main()

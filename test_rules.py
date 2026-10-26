import tempfile
import unittest
from pathlib import Path

from agent import build_messages
from memory import Memory
from rules import RULE_MAX_CHARS, load_rules


class TestRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / 'root'
        self.home = base / 'home'
        self.root.mkdir()
        self.home.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_home_then_root_agent_md(self):
        (self.home / 'AGENT.md').write_text('HOME')
        (self.root / 'AGENT.md').write_text('ROOT')
        self.assertEqual(load_rules(self.root, 'ollama', self.home), 'HOME\n\nROOT')

    def test_home_only(self):
        (self.home / 'AGENT.md').write_text('HOME')
        self.assertEqual(load_rules(self.root, 'ollama', self.home), 'HOME')

    def test_root_agent_md(self):
        (self.root / 'AGENT.md').write_text('ROOT')
        self.assertEqual(load_rules(self.root, 'ollama', self.home), 'ROOT')

    def test_claude_uses_claude_md(self):
        (self.root / 'CLAUDE.md').write_text('C')
        (self.root / 'AGENTS.md').write_text('A')
        self.assertEqual(load_rules(self.root, 'claude', self.home), 'C')
        self.assertEqual(load_rules(self.root, 'anthropic', self.home), 'C')

    def test_ollama_uses_agents_md(self):
        (self.root / 'CLAUDE.md').write_text('C')
        (self.root / 'AGENTS.md').write_text('A')
        self.assertEqual(load_rules(self.root, 'ollama', self.home), 'A')

    def test_agent_md_beats_claude_md(self):
        (self.root / 'AGENT.md').write_text('AG')
        (self.root / 'CLAUDE.md').write_text('C')
        self.assertEqual(load_rules(self.root, 'claude', self.home), 'AG')

    def test_truncation(self):
        (self.root / 'AGENT.md').write_text('x' * (RULE_MAX_CHARS + 50))
        out = load_rules(self.root, 'ollama', self.home)
        self.assertEqual(out, 'x' * RULE_MAX_CHARS + '\n[rules truncated]')

    def test_empty(self):
        self.assertEqual(load_rules(self.root, 'ollama', self.home), '')

    def test_symlink_escape_skipped(self):
        outside = Path(self.tmp.name) / 'secret.md'
        outside.write_text('SECRET')
        (self.root / 'AGENT.md').symlink_to(outside)
        self.assertEqual(load_rules(self.root, 'ollama', self.home), '')

    def test_build_messages_includes_rules(self):
        mem = Memory(Path(self.tmp.name) / 'm.db')
        try:
            text = build_messages('goal', mem, rules='RULE-X')[0]['content']
            self.assertIn('never override safety limits', text)
            self.assertTrue(text.endswith('RULE-X'))
            self.assertNotIn('User rules', build_messages('goal', mem)[0]['content'])
        finally:
            mem.close()


if __name__ == '__main__':
    unittest.main()

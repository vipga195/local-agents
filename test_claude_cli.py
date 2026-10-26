import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent import run, CLAUDE_PROMPT
from claude_cli import ClaudeCli
from memory import Memory
from test_agent import FakeBrain, tool_msg, text_msg
from tools import ToolBox


def proc(stdout='', stderr='', code=0):
    return MagicMock(stdout=stdout, stderr=stderr, returncode=code)


class TestClaudeCli(unittest.TestCase):
    def test_argv_and_kwargs(self):
        cli = ClaudeCli('/tmp/root', timeout=42)
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'k', 'ANTHROPIC_AUTH_TOKEN': 't', 'KEEP': '1'}), \
                patch('claude_cli.subprocess.run', return_value=proc(' hi \n')) as run_:
            self.assertEqual(cli.ask('q'), 'hi')
        argv = run_.call_args.args[0]
        self.assertEqual(argv, ['claude', '-p', '--tools', 'Read,Grep,Glob', '--permission-mode', 'dontAsk',
                                '--no-session-persistence', '--output-format', 'text'])
        self.assertNotIn('--bare', argv)
        kw = run_.call_args.kwargs
        self.assertEqual(kw['cwd'], '/tmp/root')
        self.assertEqual(kw['input'], 'q')
        self.assertEqual(kw['timeout'], 42)
        self.assertTrue(kw['text'])
        self.assertTrue(kw['capture_output'])
        self.assertNotIn('ANTHROPIC_API_KEY', kw['env'])
        self.assertNotIn('ANTHROPIC_AUTH_TOKEN', kw['env'])
        self.assertEqual(kw['env']['KEEP'], '1')

    def test_truncation(self):
        cli = ClaudeCli('.', max_output=10)
        with patch('claude_cli.subprocess.run', return_value=proc('x' * 50)):
            self.assertEqual(cli.ask('q'), 'x' * 10 + '\n[output truncated]')

    def test_call_limit(self):
        cli = ClaudeCli('.', max_calls=3)
        with patch('claude_cli.subprocess.run', return_value=proc('ok')) as run_:
            for _ in range(3):
                self.assertEqual(cli.ask('q'), 'ok')
            self.assertEqual(cli.ask('q'), 'error: ask_claude limit reached for this run')
        self.assertEqual(run_.call_count, 3)

    def test_not_found(self):
        with patch('claude_cli.subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(ClaudeCli('.').ask('q'), 'error: claude CLI not found')

    def test_timeout(self):
        with patch('claude_cli.subprocess.run', side_effect=subprocess.TimeoutExpired('claude', 5)):
            self.assertEqual(ClaudeCli('.', timeout=5).ask('q'), 'error: claude timed out after 5s')

    def test_nonzero_exit(self):
        with patch('claude_cli.subprocess.run', return_value=proc(stderr='e' * 500, code=2)):
            out = ClaudeCli('.').ask('q')
        self.assertEqual(out, 'error: claude exited with code 2: ' + 'e' * 300)

    def test_empty_prompt(self):
        with patch('claude_cli.subprocess.run') as run_:
            self.assertEqual(ClaudeCli('.').ask('  '), 'error: prompt is empty')
        run_.assert_not_called()


class TestClaudeToolBox(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.memory = Memory(Path(self.dir) / 'm.db')

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.dir)

    def names(self, tb):
        return [s['function']['name'] for s in tb.specs()]

    def test_spec_only_when_provided(self):
        self.assertNotIn('ask_claude', self.names(ToolBox(self.memory, self.dir)))
        tb = ToolBox(self.memory, self.dir, claude=MagicMock())
        self.assertIn('ask_claude', self.names(tb))
        self.assertTrue(tb.has_claude)

    def test_dispatch(self):
        claude = MagicMock()
        claude.ask.return_value = 'answer'
        tb = ToolBox(self.memory, self.dir, claude=claude)
        self.assertEqual(tb.call('ask_claude', {'prompt': 'q'}), 'answer')
        claude.ask.assert_called_once_with('q')

    def test_missing_prompt(self):
        tb = ToolBox(self.memory, self.dir, claude=MagicMock())
        self.assertEqual(tb.call('ask_claude', {}), 'error: missing argument prompt')

    def test_unknown_when_disabled(self):
        tb = ToolBox(self.memory, self.dir)
        self.assertEqual(tb.call('ask_claude', {'prompt': 'q'}), 'error: unknown tool ask_claude')

    def test_prompt_only_when_enabled(self):
        for claude, expected in ((MagicMock(), True), (None, False)):
            brain = FakeBrain([text_msg('done')])
            tb = ToolBox(self.memory, self.dir, claude=claude)
            run('g', brain, self.memory, tb)
            self.assertEqual(CLAUDE_PROMPT in brain.calls[0][0][0]['content'], expected)


if __name__ == '__main__':
    unittest.main()

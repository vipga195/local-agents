import os
import subprocess


class ClaudeCli:
    """Read-only bridge to the locally installed Claude Code CLI (subscription login)."""

    def __init__(self, root, timeout=300, max_calls=3, max_output=20000, binary='claude'):
        self.root = root
        self.timeout = timeout
        self.max_calls = max_calls
        self.max_output = max_output
        self.binary = binary
        self.calls = 0

    def ask(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            return 'error: prompt is empty'
        if self.calls >= self.max_calls:
            return 'error: ask_claude limit reached for this run'
        self.calls += 1
        # Drop API credentials so the subscription login is used and no API billing happens
        env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')}
        cmd = [self.binary, '-p', '--tools', 'Read,Grep,Glob', '--permission-mode', 'dontAsk',
               '--no-session-persistence', '--output-format', 'text']
        try:
            proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                                  cwd=self.root, timeout=self.timeout, env=env)
        except FileNotFoundError:
            return 'error: claude CLI not found'
        except subprocess.TimeoutExpired:
            return f'error: claude timed out after {self.timeout}s'
        except OSError as e:
            return f'error: cannot run claude: {e.strerror or type(e).__name__}'
        if proc.returncode != 0:
            return f'error: claude exited with code {proc.returncode}: {(proc.stderr or "")[:300]}'
        out = (proc.stdout or '').strip()
        if len(out) > self.max_output:
            out = out[:self.max_output] + '\n[output truncated]'
        return out

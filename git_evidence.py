import re
import subprocess
from datetime import datetime

PR_PATTERNS = (re.compile(r'\(#(\d+)\)'), re.compile(r'Merge pull request #(\d+)'))


def _git(repo, *args):
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=10)
    return r.stdout if r.returncode == 0 else None


def evidence(repo, paths, since_ts):
    """Detect whether commits touching paths since since_ts reached a remote-tracking ref. Never raises."""
    result = {'pushed': False, 'prs': []}
    try:
        if _git(repo, 'rev-parse', '--is-inside-work-tree') is None:
            return result
        since = datetime.fromtimestamp(since_ts).astimezone().isoformat()
        out = _git(repo, 'log', f'--since={since}', '--format=%H%x1f%s%x1f%b%x1e', '--', *(paths or ['.']))
        for rec in (out or '').split('\x1e'):
            parts = rec.strip().split('\x1f')
            if len(parts) < 2 or not parts[0]:
                continue
            if not (_git(repo, 'branch', '-r', '--contains', parts[0]) or '').strip():
                continue
            result['pushed'] = True
            for text in parts[1:]:
                for pat in PR_PATTERNS:
                    for m in pat.findall(text):
                        if int(m) not in result['prs']:
                            result['prs'].append(int(m))
    except Exception:
        return {'pushed': False, 'prs': []}
    return result

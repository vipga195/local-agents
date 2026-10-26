"""Load user rule files (AGENT.md / CLAUDE.md / AGENTS.md) for the system prompt."""

import os
from pathlib import Path

RULE_MAX_CHARS = 6000


def _read(path):
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None


def _inside(path, root):
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (OSError, RuntimeError):
        return False


def load_rules(root, provider, home=None) -> str:
    """Return combined rules text, truncated to RULE_MAX_CHARS ('' if none)."""
    home = Path(home if home is not None else os.path.expanduser('~/.local-agent'))
    root = Path(root)
    parts = []
    for base, trusted in ((home, True), (root, False)):
        path = base / 'AGENT.md'
        if not path.is_file() or (not trusted and not _inside(path, root)):
            continue
        text = _read(path)
        if text is not None:
            parts.append(text)
    if not parts and not any((p / 'AGENT.md').exists() for p in (home, root)):
        name = 'CLAUDE.md' if provider in ('claude', 'anthropic') else 'AGENTS.md'
        path = root / name
        if path.is_file() and _inside(path, root):
            text = _read(path)
            if text is not None:
                parts.append(text)
    text = '\n\n'.join(parts)
    if len(text) > RULE_MAX_CHARS:
        text = text[:RULE_MAX_CHARS] + '\n[rules truncated]'
    return text

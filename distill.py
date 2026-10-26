import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path
import git_evidence
from memory import Memory


def slugify(name):
    text = unicodedata.normalize('NFKD', name.lower().replace('đ', 'd'))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '-', text).strip('-')


def adr_prs(n):
    """Return PR list for an accepted adr ([] when no pushed commit is found), None if not accepted.

    Pushed commits touching the adr paths fill prs (preferred evidence); proposed adrs are never exported.
    """
    if n['status'] != 'accepted':
        return None
    d = n['data']
    ev = git_evidence.evidence(d.get('repo') or '', d.get('paths') or [], n['created'] or 0)
    return ev['prs'] if ev['pushed'] else []


def render_adr(n, project, component):
    d = n['data']
    project = d.get('project') or project
    component = d.get('component') or component
    lines = ['---', f'project: {project}', f'component: {component}', f"type: {d.get('adr_type', 'architecture')}",
             'status: accepted', f"date: {d.get('date', '')}", f"summary: {json.dumps(n['summary'], ensure_ascii=False)}",
             f"tags: [{', '.join(d.get('tags', []))}]", 'related: []', f"prs: [{', '.join(str(p) for p in n['prs'])}]",
             'environments_affected: []', '---', '', f"# {n['name']}", '',
             '## Bối cảnh', '', d.get('context', ''), '', '## Quyết định', '', d.get('decision', '')]
    if d.get('tradeoff'):
        lines += ['', '## Lý do / Trade-off', '', d['tradeoff']]
    return lines


def export_nodes(memory, out, include_proposed=False, adr_project='local-agent', adr_component='agent'):
    statuses = ('accepted', 'proposed') if include_proposed else ('accepted',)
    nodes = []
    for n in memory.list_nodes():
        if n['status'] not in statuses:
            continue
        if n['type'] == 'adr' and not n['distilled_path']:
            prs = adr_prs(n)
            if prs is None:
                continue
            n['prs'] = prs
        nodes.append(n)
    rels = {}
    for e in memory.list_edges():
        rels.setdefault((e['src_type'], e['src_name']), {}).setdefault(e['rel'], []).append(e['dst_name'])
    used = {(n['type'], Path(n['distilled_path']).stem) for n in nodes if n['distilled_path']}
    index = []
    exported = []
    for n in nodes:
        if n['distilled_path']:
            # Already exported and slimmed: keep it in the index, do not rewrite its file
            stub = Path(n['distilled_path'])
            link = stub.relative_to(out).as_posix() if stub.is_relative_to(out) else stub.as_posix()
            index.append(f"- [{n['type']}:{n['name']}]({link}) - {n['summary']}")
            continue
        slug = slugify(n['name']) or f"node-{n['id']}"
        if n['type'] == 'adr':
            slug = f"{n['data'].get('date', '')}-{slug}"
        if (n['type'], slug) in used:
            slug = f"{slug}-{n['id']}"
        used.add((n['type'], slug))
        if n['type'] == 'adr':
            path = out / 'adr' / f'{slug}.md'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(render_adr(n, adr_project, adr_component)) + '\n', encoding='utf-8')
            exported.append(dict(n, path=path.resolve()))
            index.append(f"- [adr:{n['name']}](adr/{slug}.md) - {n['summary']}")
            continue
        lines = ['---', f"type: {n['type']}", f"name: {json.dumps(n['name'], ensure_ascii=False)}", f"status: {n['status']}"]
        for rel, targets in rels.get((n['type'], n['name']), {}).items():
            lines.append(f"{rel}: [{', '.join(json.dumps(t, ensure_ascii=False) for t in targets)}]")
        lines += ['---', '', f"# {n['name']}", '', n['summary']]
        if n['data']:
            lines += ['', '## Data', '', '```json', json.dumps(n['data'], indent=2, ensure_ascii=False), '```']
        path = out / n['type'] / f'{slug}.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        exported.append(dict(n, path=path.resolve()))
        index.append(f"- [{n['type']}:{n['name']}]({n['type']}/{slug}.md) - {n['summary']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return exported


def is_kb_path(out):
    return 'knowledge-base' in Path(out).expanduser().resolve().parts


def slim_exported(memory, exported):
    """Slim exported accepted nodes (clear data, store file path); keeps nodes and edges. Returns count."""
    eligible = [n for n in exported if n['status'] == 'accepted']
    if not eligible:
        return 0
    for n in eligible:
        if not n['path'].is_file() or n['path'].stat().st_size == 0:
            raise RuntimeError(f"export verification failed for node #{n['id']}: {n['path']}")
    db_file = next(r['file'] for r in memory.conn.execute('PRAGMA database_list') if r['name'] == 'main')
    backup = sqlite3.connect(db_file + '.bak')
    try:
        memory.conn.backup(backup)
    finally:
        backup.close()
    return memory.slim_nodes([(n['id'], str(n['path'])) for n in eligible])


def distill_session(memory, out_dir):
    """Export accepted nodes and slim them in the DB. Never raises; returns a status line or None."""
    try:
        if not any(n['type'] != 'adr' or adr_prs(n) is not None for n in memory.list_nodes(status='accepted') if not n['distilled_path']):
            return None
        out = Path(out_dir).expanduser().resolve()
        if is_kb_path(out):
            print('warning: distill skipped, refusing to export inside a knowledge-base directory', file=sys.stderr)
            return None
        exported = export_nodes(memory, out)
        slimmed = slim_exported(memory, exported)
        return f'distilled {len(exported)} nodes to {out}, slimmed {slimmed}'
    except Exception as e:
        print(f'warning: distill failed: {e}', file=sys.stderr)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Review and export agent memory nodes as markdown.')
    parser.add_argument('--db', default=os.environ.get('AGENT_DB') or '~/.local-agent/memory.db', help='Database path')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_list = sub.add_parser('list')
    p_list.add_argument('--status', choices=['proposed', 'accepted', 'rejected'])
    sub.add_parser('approve').add_argument('id', type=int)
    sub.add_parser('reject').add_argument('id', type=int)
    p_exp = sub.add_parser('export')
    p_exp.add_argument('--out', required=True)
    p_exp.add_argument('--include-proposed', action='store_true')
    p_exp.add_argument('--adr-project', default='local-agent', help='Default KB project for exported adr nodes')
    p_exp.add_argument('--adr-component', default='agent', help='Default KB component for exported adr nodes')
    p_exp.add_argument('--slim', action='store_true', help='Slim exported accepted nodes in the database (clear data, keep summary/edges, store file path)')
    args = parser.parse_args(argv)
    if args.cmd == 'export':
        out = Path(args.out).expanduser().resolve()
        if is_kb_path(out):
            print('refusing to export inside a knowledge-base directory', file=sys.stderr)
            return 2
    db = Path(args.db).expanduser()
    db.parent.mkdir(parents=True, exist_ok=True)
    memory = Memory(db)
    try:
        if args.cmd == 'list':
            for n in memory.list_nodes(status=args.status):
                print(f"#{n['id']} [{n['status']}] {n['type']}:{n['name']} - {n['summary']}")
            return 0
        if args.cmd == 'export':
            exported = export_nodes(memory, out, args.include_proposed, args.adr_project, args.adr_component)
            print(f'exported {len(exported)} nodes to {out}')
            if args.slim:
                try:
                    slimmed = slim_exported(memory, exported)
                except RuntimeError as e:
                    print(f'slim aborted, nothing changed: {e}', file=sys.stderr)
                    return 1
                print(f'slimmed {slimmed} nodes')
            return 0
        status = 'accepted' if args.cmd == 'approve' else 'rejected'
        try:
            memory.set_node_status(args.id, status)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f'node #{args.id} -> {status}')
        return 0
    finally:
        memory.close()


if __name__ == '__main__':
    sys.exit(main())

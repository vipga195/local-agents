import argparse
import os
import sys
from pathlib import Path
from memory import Memory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Review agent lessons.')
    parser.add_argument('--db', default=os.environ.get('AGENT_DB') or '~/.local-agent/memory.db', help='Database path')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_list = sub.add_parser('list')
    p_list.add_argument('--status', choices=['proposed', 'accepted', 'rejected'])
    p_list.add_argument('--limit', type=int, default=50)
    sub.add_parser('approve').add_argument('id', type=int)
    sub.add_parser('reject').add_argument('id', type=int)
    args = parser.parse_args(argv)
    db = Path(args.db).expanduser()
    db.parent.mkdir(parents=True, exist_ok=True)
    memory = Memory(db)
    try:
        if args.cmd == 'list':
            for l in memory.list_lessons(status=args.status, limit=args.limit):
                print(f"#{l['id']} [{l['status']}] conf={l['confidence']:.2f} | {l['rule']}")
                print(f"    {l['situation']}")
            return 0
        status = 'accepted' if args.cmd == 'approve' else 'rejected'
        try:
            memory.set_lesson_status(args.id, status)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f'lesson #{args.id} -> {status}')
        return 0
    finally:
        memory.close()


if __name__ == '__main__':
    sys.exit(main())

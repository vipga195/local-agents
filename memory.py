import json
import re
import sqlite3
import time


class Memory:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self._create_schema()
        self.conn.commit()

    def _create_schema(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated REAL NOT NULL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                situation TEXT NOT NULL,
                rule TEXT NOT NULL,
                evidence TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'proposed' CHECK(status IN ('proposed','accepted','rejected')),
                project TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                created REAL NOT NULL,
                updated REAL NOT NULL
            )
        ''')
        self.conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(situation, rule, tags, content='lessons', content_rowid='id')
        ''')
        # External-content triggers: keep FTS index in sync with lessons
        self.conn.execute('''
            CREATE TRIGGER IF NOT EXISTS lessons_insert_fts
            AFTER INSERT ON lessons
            BEGIN
                INSERT INTO lessons_fts(rowid, situation, rule, tags) VALUES (new.id, new.situation, new.rule, new.tags);
            END
        ''')
        self.conn.execute('''
            CREATE TRIGGER IF NOT EXISTS lessons_delete_fts
            AFTER DELETE ON lessons
            BEGIN
                INSERT INTO lessons_fts(lessons_fts, rowid, situation, rule, tags) VALUES('delete', old.id, old.situation, old.rule, old.tags);
            END
        ''')
        self.conn.execute('''
            CREATE TRIGGER IF NOT EXISTS lessons_update_fts
            AFTER UPDATE ON lessons
            BEGIN
                INSERT INTO lessons_fts(lessons_fts, rowid, situation, rule, tags) VALUES('delete', old.id, old.situation, old.rule, old.tags);
                INSERT INTO lessons_fts(rowid, situation, rule, tags) VALUES (new.id, new.situation, new.rule, new.tags);
            END
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL,
                started REAL NOT NULL,
                finished REAL,
                outcome TEXT DEFAULT '',
                steps TEXT DEFAULT '[]',
                route TEXT DEFAULT '',
                user_verdict TEXT DEFAULT '',
                title TEXT NOT NULL DEFAULT ''
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                hits INTEGER NOT NULL,
                created REAL NOT NULL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT DEFAULT '',
                data TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'proposed' CHECK(status IN ('proposed','accepted','rejected')),
                created REAL,
                updated REAL,
                UNIQUE(type, name)
            )
        ''')
        # Migrate DBs created before nodes.status existed (ALTER cannot add a CHECK; validated in Python)
        if 'status' not in [r['name'] for r in self.conn.execute('PRAGMA table_info(nodes)')]:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN status TEXT NOT NULL DEFAULT 'proposed'")
        if 'distilled_path' not in [r['name'] for r in self.conn.execute('PRAGMA table_info(nodes)')]:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN distilled_path TEXT NOT NULL DEFAULT ''")
        self.conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(name, summary, data, content='nodes', content_rowid='id')
        ''')
        self.conn.execute('''
            CREATE TRIGGER IF NOT EXISTS nodes_insert_fts
            AFTER INSERT ON nodes
            BEGIN
                INSERT INTO nodes_fts(rowid, name, summary, data) VALUES (new.id, new.name, new.summary, new.data);
            END
        ''')
        self.conn.execute('''
            CREATE TRIGGER IF NOT EXISTS nodes_delete_fts
            AFTER DELETE ON nodes
            BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, name, summary, data) VALUES('delete', old.id, old.name, old.summary, old.data);
            END
        ''')
        self.conn.execute('''
            CREATE TRIGGER IF NOT EXISTS nodes_update_fts
            AFTER UPDATE ON nodes
            BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, name, summary, data) VALUES('delete', old.id, old.name, old.summary, old.data);
                INSERT INTO nodes_fts(rowid, name, summary, data) VALUES (new.id, new.name, new.summary, new.data);
            END
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                src INTEGER NOT NULL,
                rel TEXT NOT NULL,
                dst INTEGER NOT NULL,
                UNIQUE(src, rel, dst)
            )
        ''')
        # Migrate DBs created before runs.title existed
        if 'title' not in [r['name'] for r in self.conn.execute('PRAGMA table_info(runs)')]:
            self.conn.execute("ALTER TABLE runs ADD COLUMN title TEXT NOT NULL DEFAULT ''")


    def set_profile(self, key, value):
        updated = time.time()
        self.conn.execute('''
            INSERT INTO profile (key, value, updated)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated
        ''', (key, value, updated))
        self.conn.commit()

    def get_profile(self, key):
        row = self.conn.execute('SELECT value FROM profile WHERE key=?', (key,)).fetchone()
        return row['value'] if row else None

    def all_profile(self):
        return {row['key']: row['value'] for row in self.conn.execute('SELECT key, value FROM profile')}

    def add_lesson(self, situation, rule, evidence='', confidence=0.5, project='', tags=''):
        created = updated = time.time()
        cursor = self.conn.execute('''
            INSERT INTO lessons (situation, rule, evidence, confidence, status, project, tags, created, updated)
            VALUES (?, ?, ?, ?, 'proposed', ?, ?, ?, ?)
        ''', (situation, rule, evidence, confidence, project, tags, created, updated))
        self.conn.commit()
        return cursor.lastrowid

    def set_lesson_status(self, lesson_id, status):
        if status not in ('proposed', 'accepted', 'rejected'):
            raise ValueError(f'Invalid status: {status}')
        cursor = self.conn.execute('''
            UPDATE lessons SET status=?, updated=? WHERE id=?
        ''', (status, time.time(), lesson_id))
        if cursor.rowcount == 0:
            raise ValueError(f'Unknown lesson ID: {lesson_id}')
        self.conn.commit()

    def search_lessons(self, query, project='', limit=5, include_proposed=True):
        tokens = re.findall(r'\w+', query, re.UNICODE)
        if not tokens:
            return []
        match = ' OR '.join(f'"{t}"' for t in tokens)
        params = [match]
        sql = '''
            SELECT l.id, l.situation, l.rule, l.evidence, l.confidence, l.status, l.project, l.tags
            FROM lessons_fts
            JOIN lessons l ON l.id = lessons_fts.rowid
            WHERE lessons_fts MATCH ? AND l.status != 'rejected'
        '''
        if not include_proposed:
            sql += " AND l.status = 'accepted'"
        if project:
            sql += " AND (l.project = ? OR l.project = '')"
            params.append(project)
        sql += ' ORDER BY bm25(lessons_fts) LIMIT ?'
        params.append(limit)
        results = [dict(row) for row in self.conn.execute(sql, params)]
        self.conn.execute('''
            INSERT INTO searches (query, hits, created)
            VALUES (?, ?, ?)
        ''', (query, len(results), time.time()))
        self.conn.commit()
        return results

    def list_lessons(self, status=None, limit=50):
        sql = 'SELECT id, situation, rule, evidence, confidence, status, project, tags, created FROM lessons'
        params = []
        if status:
            sql += ' WHERE status = ?'
            params.append(status)
        sql += ' ORDER BY id DESC LIMIT ?'
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params)]

    def _node_id(self, type, name):
        row = self.conn.execute('SELECT id FROM nodes WHERE type=? AND name=?', (type.strip(), name.strip())).fetchone()
        return row['id'] if row else None

    def add_node(self, type, name, summary='', data=None):
        type, name = (type or '').strip(), (name or '').strip()
        if not type or not name:
            raise ValueError('type and name must be non-empty')
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError('data must be an object')
        row = self.conn.execute('SELECT distilled_path FROM nodes WHERE type=? AND name=?', (type, name)).fetchone()
        if row and row['distilled_path']:
            raise ValueError('node is distilled (approved); edit its file instead of add_node')
        now = time.time()
        self.conn.execute('''
            INSERT INTO nodes (type, name, summary, data, created, updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(type, name) DO UPDATE SET
                summary=CASE WHEN excluded.summary != '' THEN excluded.summary ELSE nodes.summary END,
                data=CASE WHEN excluded.data != '{}' THEN excluded.data ELSE nodes.data END,
                updated=excluded.updated
        ''', (type, name, summary, json.dumps(data, ensure_ascii=False), now, now))
        self.conn.commit()
        return self._node_id(type, name)

    def link(self, src_type, src_name, rel, dst_type, dst_name):
        rel = (rel or '').strip()
        if not rel:
            raise ValueError('rel must be non-empty')
        src = self._node_id(src_type, src_name)
        dst = self._node_id(dst_type, dst_name)
        if src is None or dst is None:
            raise ValueError('unknown node')
        self.conn.execute('INSERT OR IGNORE INTO edges (src, rel, dst) VALUES (?, ?, ?)', (src, rel, dst))
        self.conn.commit()

    def find_nodes(self, query, type='', limit=5):
        tokens = re.findall(r'\w+', query, re.UNICODE)
        if not tokens:
            return []
        match = ' OR '.join(f'"{t}"' for t in tokens)
        params = [match]
        sql = '''
            SELECT n.id, n.type, n.name, n.summary, n.status, n.distilled_path
            FROM nodes_fts
            JOIN nodes n ON n.id = nodes_fts.rowid
            WHERE nodes_fts MATCH ?
        '''
        if type:
            sql += ' AND n.type = ?'
            params.append(type)
        sql += ' ORDER BY bm25(nodes_fts) LIMIT ?'
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params)]

    def get_node(self, type, name):
        row = self.conn.execute('SELECT id, type, name, summary, data, status, distilled_path FROM nodes WHERE type=? AND name=?', (type.strip(), name.strip())).fetchone()
        if row is None:
            return None
        node = dict(row)
        node['data'] = json.loads(node['data'])
        node['out'] = [dict(r) for r in self.conn.execute('''
            SELECT e.rel, n.type, n.name FROM edges e JOIN nodes n ON n.id = e.dst WHERE e.src=? ORDER BY e.rel, n.name
        ''', (node['id'],))]
        node['in'] = [dict(r) for r in self.conn.execute('''
            SELECT e.rel, n.type, n.name FROM edges e JOIN nodes n ON n.id = e.src WHERE e.dst=? ORDER BY e.rel, n.name
        ''', (node['id'],))]
        return node

    def set_node_status(self, node_id, status):
        if status not in ('proposed', 'accepted', 'rejected'):
            raise ValueError(f'Invalid status: {status}')
        cursor = self.conn.execute('UPDATE nodes SET status=?, updated=? WHERE id=?', (status, time.time(), node_id))
        if cursor.rowcount == 0:
            raise ValueError(f'Unknown node ID: {node_id}')
        self.conn.commit()

    def list_nodes(self, status=None):
        sql = 'SELECT id, type, name, summary, data, status, distilled_path, created FROM nodes'
        params = []
        if status:
            sql += ' WHERE status = ?'
            params.append(status)
        sql += ' ORDER BY id'
        nodes = []
        for row in self.conn.execute(sql, params):
            node = dict(row)
            node['data'] = json.loads(node['data'])
            nodes.append(node)
        return nodes

    def slim_nodes(self, items):
        items = list(items)
        with self.conn:
            for node_id, path in items:
                self.conn.execute("UPDATE nodes SET distilled_path=?, data='{}', updated=? WHERE id=?", (str(path), time.time(), node_id))
        return len(items)

    def list_edges(self):
        return [dict(r) for r in self.conn.execute('''
            SELECT s.type AS src_type, s.name AS src_name, e.rel, d.type AS dst_type, d.name AS dst_name
            FROM edges e JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst
            ORDER BY e.src, e.rel, e.dst
        ''')]

    def start_run(self, goal):
        cursor = self.conn.execute('''
            INSERT INTO runs (goal, started)
            VALUES (?, ?)
        ''', (goal, time.time()))
        self.conn.commit()
        return cursor.lastrowid

    def log_step(self, run_id, step):
        row = self.conn.execute('SELECT steps FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError(f'Unknown run ID: {run_id}')
        steps = json.loads(row['steps'])
        steps.append(step)
        self.conn.execute('UPDATE runs SET steps=? WHERE id=?', (json.dumps(steps), run_id))
        self.conn.commit()

    def finish_run(self, run_id, outcome):
        self.conn.execute('''
            UPDATE runs SET finished=?, outcome=? WHERE id=?
        ''', (time.time(), outcome, run_id))
        self.conn.commit()

    def set_verdict(self, run_id, verdict):
        self.conn.execute('UPDATE runs SET user_verdict=? WHERE id=?', (verdict, run_id))
        self.conn.commit()

    def set_title(self, run_id, title):
        cursor = self.conn.execute('UPDATE runs SET title=? WHERE id=?', (title, run_id))
        if cursor.rowcount == 0:
            raise ValueError(f'Unknown run ID: {run_id}')
        self.conn.commit()

    def add_route(self, run_id, note):
        row = self.conn.execute('SELECT route FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError(f'Unknown run ID: {run_id}')
        route = row['route']
        route = route + '; ' + note if route else note
        self.conn.execute('UPDATE runs SET route=? WHERE id=?', (route, run_id))
        self.conn.commit()

    def get_run(self, run_id):
        row = self.conn.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        run['steps'] = json.loads(run['steps'])
        return run

    def empty_search_count(self):
        return self.conn.execute('SELECT COUNT(*) FROM searches WHERE hits = 0').fetchone()[0]

    def close(self):
        self.conn.close()

import os
import unittest
from tempfile import TemporaryDirectory

from memory import Memory


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.db_path = os.path.join(self.dir.name, 'test.db')
        self.memory = Memory(self.db_path)

    def tearDown(self):
        self.memory.close()
        self.dir.cleanup()

    def test_schema_idempotent(self):
        lesson_id = self.memory.add_lesson('alpha', 'bravo')
        self.memory.close()
        self.memory = Memory(self.db_path)
        lessons = self.memory.search_lessons('alpha')
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]['id'], lesson_id)

    def test_add_lesson_defaults_proposed(self):
        lesson_id = self.memory.add_lesson('alpha', 'bravo')
        lessons = self.memory.search_lessons('alpha')
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]['id'], lesson_id)
        self.assertEqual(lessons[0]['status'], 'proposed')

    def test_search_by_situation_rule_tags(self):
        situation_id = self.memory.add_lesson('alpha', 'x1')
        rule_id = self.memory.add_lesson('y1', 'bravo')
        tags_id = self.memory.add_lesson('z1', 'w1', tags='charlie')
        self.assertEqual([r['id'] for r in self.memory.search_lessons('alpha')], [situation_id])
        self.assertEqual([r['id'] for r in self.memory.search_lessons('bravo')], [rule_id])
        self.assertEqual([r['id'] for r in self.memory.search_lessons('charlie')], [tags_id])

    def test_search_vietnamese_diacritics(self):
        lesson_id = self.memory.add_lesson('cảnh đêm', 'dùng ánh sáng ấm cho cảnh đêm')
        lessons = self.memory.search_lessons('ánh sáng')
        self.assertEqual([r['id'] for r in lessons], [lesson_id])

    def test_search_punctuation_no_error(self):
        self.memory.add_lesson('alpha', 'bravo')
        lessons = self.memory.search_lessons('a"b OR (c')
        self.assertIsInstance(lessons, list)

    def test_search_no_tokens_returns_empty(self):
        self.memory.add_lesson('alpha', 'bravo')
        self.assertEqual(self.memory.search_lessons('!!! ???'), [])

    def test_rejected_excluded(self):
        lesson_id = self.memory.add_lesson('alpha', 'bravo')
        self.memory.set_lesson_status(lesson_id, 'rejected')
        self.assertEqual(self.memory.search_lessons('alpha'), [])

    def test_include_proposed_false(self):
        self.memory.add_lesson('alpha', 'bravo')
        accepted_id = self.memory.add_lesson('alpha', 'delta')
        self.memory.set_lesson_status(accepted_id, 'accepted')
        lessons = self.memory.search_lessons('alpha', include_proposed=False)
        self.assertEqual([r['id'] for r in lessons], [accepted_id])
        self.assertEqual(lessons[0]['status'], 'accepted')

    def test_project_filter(self):
        p1_id = self.memory.add_lesson('alpha', 'bravo', project='p1')
        global_id = self.memory.add_lesson('alpha', 'bravo', project='')
        self.memory.add_lesson('alpha', 'bravo', project='p2')
        ids = {r['id'] for r in self.memory.search_lessons('alpha', project='p1')}
        self.assertEqual(ids, {p1_id, global_id})

    def test_set_status_invalid_raises(self):
        lesson_id = self.memory.add_lesson('alpha', 'bravo')
        with self.assertRaises(ValueError):
            self.memory.set_lesson_status(lesson_id, 'bogus')
        with self.assertRaises(ValueError):
            self.memory.set_lesson_status(9999, 'accepted')

    def test_empty_search_logged(self):
        self.assertEqual(self.memory.empty_search_count(), 0)
        self.assertEqual(self.memory.search_lessons('nomatchword'), [])
        self.assertEqual(self.memory.empty_search_count(), 1)

    def test_run_lifecycle(self):
        run_id = self.memory.start_run('goal1')
        self.memory.log_step(run_id, {'action': 'a'})
        self.memory.log_step(run_id, {'action': 'b'})
        self.memory.finish_run(run_id, 'ok')
        self.memory.set_verdict(run_id, 'good')
        run = self.memory.get_run(run_id)
        self.assertEqual(run['goal'], 'goal1')
        self.assertEqual(run['steps'], [{'action': 'a'}, {'action': 'b'}])
        self.assertEqual(run['outcome'], 'ok')
        self.assertEqual(run['user_verdict'], 'good')
        self.assertIsNotNone(run['finished'])

    def test_profile_roundtrip(self):
        self.memory.set_profile('name', 'x')
        self.memory.set_profile('name', 'y')
        self.assertEqual(self.memory.get_profile('name'), 'y')
        self.assertIsNone(self.memory.get_profile('missing'))
        self.memory.set_profile('lang', 'vi')
        self.assertEqual(self.memory.all_profile(), {'name': 'y', 'lang': 'vi'})

    def test_list_lessons(self):
        a = self.memory.add_lesson('s1', 'r1')
        b = self.memory.add_lesson('s2', 'r2')
        c = self.memory.add_lesson('s3', 'r3')
        self.memory.set_lesson_status(b, 'accepted')
        self.assertEqual([l['id'] for l in self.memory.list_lessons()], [c, b, a])
        self.assertEqual([l['id'] for l in self.memory.list_lessons(status='accepted')], [b])
        self.assertEqual(len(self.memory.list_lessons(limit=2)), 2)
        self.assertIn('created', self.memory.list_lessons()[0])

    def test_node_upsert_keeps_id(self):
        a = self.memory.add_node('script', 'Thạch Sanh', 'v1', {'k': 1})
        b = self.memory.add_node('script', 'Thạch Sanh', 'v2', {'k': 2})
        self.assertEqual(a, b)
        node = self.memory.get_node('script', 'Thạch Sanh')
        self.assertEqual((node['summary'], node['data']), ('v2', {'k': 2}))
        self.assertEqual(len(self.memory.find_nodes('v1')), 0)

    def test_node_validation(self):
        with self.assertRaises(ValueError):
            self.memory.add_node('script', 'x', data=['a'])
        with self.assertRaises(ValueError):
            self.memory.add_node(' ', 'x')
        with self.assertRaises(ValueError):
            self.memory.add_node('script', '')

    def test_link_idempotent_and_unknown(self):
        self.memory.add_node('script', 's')
        self.memory.add_node('character', 'c')
        self.memory.link('script', 's', 'has_character', 'character', 'c')
        self.memory.link('script', 's', 'has_character', 'character', 'c')
        self.assertEqual(len(self.memory.get_node('script', 's')['out']), 1)
        with self.assertRaises(ValueError):
            self.memory.link('script', 's', 'x', 'character', 'missing')

    def test_find_nodes_vietnamese(self):
        self.memory.add_node('script', 'Thạch Sanh', 'truyện cổ tích')
        self.memory.add_node('character', 'Lý Thông', 'nhân vật phản diện')
        self.assertEqual([n['name'] for n in self.memory.find_nodes('thạch sanh')], ['Thạch Sanh'])
        self.assertEqual(self.memory.find_nodes('thạch sanh', type='character'), [])
        self.assertEqual(self.memory.find_nodes('!!!'), [])

    def test_get_node_in_out(self):
        self.memory.add_node('script', 's')
        self.memory.add_node('character', 'c')
        self.memory.link('script', 's', 'has_character', 'character', 'c')
        self.assertEqual(self.memory.get_node('script', 's')['out'], [{'rel': 'has_character', 'type': 'character', 'name': 'c'}])
        self.assertEqual(self.memory.get_node('character', 'c')['in'], [{'rel': 'has_character', 'type': 'script', 'name': 's'}])
        self.assertIsNone(self.memory.get_node('script', 'nope'))

    def test_set_node_status_and_list(self):
        a = self.memory.add_node('script', 's', 'sum', {'k': 1})
        self.memory.add_node('character', 'c')
        self.memory.set_node_status(a, 'accepted')
        self.assertEqual([n['name'] for n in self.memory.list_nodes(status='accepted')], ['s'])
        self.assertEqual(self.memory.list_nodes()[0]['data'], {'k': 1})
        self.assertEqual(len(self.memory.list_nodes()), 2)
        with self.assertRaises(ValueError):
            self.memory.set_node_status(a, 'bad')
        with self.assertRaises(ValueError):
            self.memory.set_node_status(999, 'accepted')

    def test_list_edges(self):
        self.memory.add_node('script', 's')
        self.memory.add_node('character', 'c')
        self.memory.link('script', 's', 'has_character', 'character', 'c')
        self.assertEqual(self.memory.list_edges(), [{'src_type': 'script', 'src_name': 's', 'rel': 'has_character', 'dst_type': 'character', 'dst_name': 'c'}])


if __name__ == '__main__':
    unittest.main()

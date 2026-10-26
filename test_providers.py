import io
import json
import unittest
import urllib.error
from unittest import mock

from brain import AnthropicBrain, BrainError, OpenAIBrain

KEY = 'sk-secret-123'
TOOL = {'type': 'function', 'function': {'name': 'f', 'description': 'd', 'parameters': {'type': 'object', 'properties': {}}}}
HISTORY = [
    {'role': 'system', 'content': 'S1'},
    {'role': 'system', 'content': 'S2'},
    {'role': 'user', 'content': 'go'},
    {'role': 'assistant', 'content': 'ok', 'tool_calls': [
        {'function': {'name': 'a', 'arguments': {'x': 1}}},
        {'function': {'name': 'b', 'arguments': {}}}]},
    {'role': 'tool', 'content': 'ra', 'tool_name': 'a'},
    {'role': 'tool', 'content': 'rb', 'tool_name': 'b'},
]


class FakeResp:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(body=None, side_effect=None):
    m = mock.Mock(side_effect=side_effect) if side_effect else mock.Mock(return_value=FakeResp(body))
    return mock.patch('urllib.request.urlopen', m), m


def sent(m):
    req = m.call_args[0][0]
    return req, json.loads(req.data)


class Common:
    def call(self, body=None, side_effect=None, messages=None, tools=None):
        p, m = patch_urlopen(body, side_effect)
        with p:
            try:
                return self.brain.chat(messages or [{'role': 'user', 'content': 'x'}], tools), m
            except BrainError as e:
                e.mock = m
                raise

    def test_http_error_hides_key(self):
        err = urllib.error.HTTPError('u', 401, 'x', {}, io.BytesIO(('bad ' + KEY).encode()))
        with self.assertRaises(BrainError) as cm:
            self.call(side_effect=err)
        self.assertIn('401', str(cm.exception))
        self.assertNotIn(KEY, str(cm.exception))

    def test_url_error_and_timeout(self):
        with self.assertRaises(BrainError):
            self.call(side_effect=urllib.error.URLError('down'))
        with self.assertRaises(BrainError):
            self.call(side_effect=TimeoutError())

    def test_invalid_json(self):
        with self.assertRaises(BrainError):
            self.call(b'not json')

    def test_non_dict_json(self):
        with self.assertRaises(BrainError):
            self.call(b'[1]')


class TestOpenAI(Common, unittest.TestCase):
    def setUp(self):
        self.brain = OpenAIBrain(model='m', api_key=KEY)

    def reply(self, message):
        return {'choices': [{'message': message}]}

    def test_text_and_request(self):
        r, m = self.call(self.reply({'role': 'assistant', 'content': 'hi'}), tools=[TOOL])
        self.assertEqual(r, {'role': 'assistant', 'content': 'hi'})
        req, body = sent(m)
        self.assertEqual(req.full_url, 'https://api.openai.com/v1/chat/completions')
        self.assertEqual(req.get_header('Authorization'), 'Bearer ' + KEY)
        self.assertEqual((body['model'], body['temperature'], body['tools']), ('m', 0.2, [TOOL]))

    def test_none_content_and_host_strip(self):
        self.brain = OpenAIBrain(host='http://h/v1/', model='m', api_key=KEY)
        r, m = self.call(self.reply({'content': None}))
        self.assertEqual(r['content'], '')
        self.assertEqual(sent(m)[0].full_url, 'http://h/v1/chat/completions')
        self.assertNotIn('tools', sent(m)[1])

    def test_tool_call_reply(self):
        msg = {'content': None, 'tool_calls': [
            {'id': 'z', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"a": 1}'}},
            {'id': 'y', 'type': 'function', 'function': {'name': 'g', 'arguments': 'oops'}},
            {'id': 'w', 'type': 'function', 'function': {'name': 'h', 'arguments': '[1]'}}]}
        r, _ = self.call(self.reply(msg))
        self.assertEqual(r['tool_calls'], [
            {'function': {'name': 'f', 'arguments': {'a': 1}}},
            {'function': {'name': 'g', 'arguments': {}}},
            {'function': {'name': 'h', 'arguments': {}}}])

    def test_outgoing_conversion(self):
        _, m = self.call(self.reply({'content': 'x'}), messages=HISTORY)
        msgs = sent(m)[1]['messages']
        self.assertEqual(msgs[3]['tool_calls'], [
            {'id': 'call_0', 'type': 'function', 'function': {'name': 'a', 'arguments': '{"x": 1}'}},
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'b', 'arguments': '{}'}}])
        self.assertEqual(msgs[4], {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'ra'})
        self.assertEqual(msgs[5], {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'rb'})

    def test_missing_fields(self):
        for body in ({}, {'choices': []}, {'choices': [{}]}, {'choices': ['x']}):
            with self.assertRaises(BrainError):
                self.call(body)


class TestAnthropic(Common, unittest.TestCase):
    def setUp(self):
        self.brain = AnthropicBrain(model='m', api_key=KEY)

    def test_text_and_request(self):
        r, m = self.call({'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}, tools=[TOOL])
        self.assertEqual(r, {'role': 'assistant', 'content': 'ab'})
        req, body = sent(m)
        self.assertEqual(req.full_url, 'https://api.anthropic.com/v1/messages')
        self.assertEqual(req.get_header('X-api-key'), KEY)
        self.assertEqual(req.get_header('Anthropic-version'), '2023-06-01')
        self.assertEqual(req.get_header('Content-type'), 'application/json')
        self.assertEqual((body['model'], body['max_tokens'], body['temperature']), ('m', 4096, 0.2))
        self.assertEqual(body['tools'], [{'name': 'f', 'description': 'd', 'input_schema': {'type': 'object', 'properties': {}}}])
        self.assertNotIn('system', body)

    def test_tool_use_reply(self):
        r, _ = self.call({'content': [{'type': 'text', 'text': 't'},
                                      {'type': 'tool_use', 'id': 'i', 'name': 'f', 'input': {'a': 1}}]})
        self.assertEqual(r['content'], 't')
        self.assertEqual(r['tool_calls'], [{'function': {'name': 'f', 'arguments': {'a': 1}}}])

    def test_outgoing_conversion(self):
        _, m = self.call({'content': []}, messages=HISTORY)
        body = sent(m)[1]
        self.assertEqual(body['system'], 'S1\n\nS2')
        self.assertEqual(body['messages'], [
            {'role': 'user', 'content': 'go'},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': 'ok'},
                {'type': 'tool_use', 'id': 'call_0', 'name': 'a', 'input': {'x': 1}},
                {'type': 'tool_use', 'id': 'call_1', 'name': 'b', 'input': {}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'call_0', 'content': 'ra'},
                {'type': 'tool_result', 'tool_use_id': 'call_1', 'content': 'rb'}]}])

    def test_missing_fields(self):
        for body in ({}, {'content': 'x'}, {'content': None}):
            with self.assertRaises(BrainError):
                self.call(body)


if __name__ == '__main__':
    unittest.main()

import unittest
import http.server
import socket
import threading
import json

from brain import OllamaBrain, BrainError, tool_calls


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        try:
            self.server.last_body = json.loads(raw)
        except ValueError:
            self.server.last_body = None
        self.send_response(self.server.reply_status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(self.server.reply_body)

    def log_message(self, format, *args):
        pass


class TestBrain(unittest.TestCase):
    def setUp(self):
        # Per-test server; handler state lives on the server object
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.last_body = None
        self.server.reply_status = 200
        self.server.reply_body = json.dumps({'message': {'role': 'assistant', 'content': 'hi'}}).encode()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def test_chat_returns_message_and_sends_payload(self):
        brain = OllamaBrain(host=self.url, model='m1', timeout=5)
        msgs = [{'role': 'user', 'content': 'x'}]
        response = brain.chat(msgs)
        self.assertEqual(response, {'role': 'assistant', 'content': 'hi'})
        self.assertEqual(self.server.last_body['model'], 'm1')
        self.assertEqual(self.server.last_body['messages'], msgs)
        self.assertFalse(self.server.last_body['stream'])
        self.assertEqual(self.server.last_body['options'], {'temperature': 0.2})

    def test_tools_key_only_when_given(self):
        brain = OllamaBrain(host=self.url, model='m1', timeout=5)
        msgs = [{'role': 'user', 'content': 'x'}]
        brain.chat(msgs)
        self.assertNotIn('tools', self.server.last_body)
        tool = {'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object', 'properties': {}}}}
        brain.chat(msgs, tools=[tool])
        self.assertIn('tools', self.server.last_body)
        self.assertEqual(self.server.last_body['tools'], [tool])

    def test_custom_model_and_host_without_scheme(self):
        brain = OllamaBrain(host='127.0.0.1:%d' % self.server.server_address[1], model='other', timeout=5)
        self.assertTrue(brain.host.startswith('http://'))
        msgs = [{'role': 'user', 'content': 'x'}]
        response = brain.chat(msgs)
        self.assertEqual(response, {'role': 'assistant', 'content': 'hi'})
        self.assertEqual(self.server.last_body['model'], 'other')

    def test_http_500_raises(self):
        self.server.reply_status = 500
        self.server.reply_body = b'boom'
        brain = OllamaBrain(host=self.url, model='m1', timeout=5)
        with self.assertRaises(BrainError) as context:
            brain.chat([{'role': 'user', 'content': 'x'}])
        self.assertIn('500', str(context.exception))

    def test_non_json_body_raises(self):
        self.server.reply_body = b'not json'
        brain = OllamaBrain(host=self.url, model='m1', timeout=5)
        with self.assertRaises(BrainError):
            brain.chat([{'role': 'user', 'content': 'x'}])

    def test_missing_message_raises(self):
        self.server.reply_body = b'{"error": "x"}'
        brain = OllamaBrain(host=self.url, model='m1', timeout=5)
        with self.assertRaises(BrainError):
            brain.chat([{'role': 'user', 'content': 'x'}])

    def test_closed_port_raises(self):
        # Reserve a free port, then close it so nothing listens there
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        brain = OllamaBrain(host='http://127.0.0.1:%d' % port, model='m1', timeout=2)
        with self.assertRaises(BrainError):
            brain.chat([{'role': 'user', 'content': 'x'}])

    def test_tool_calls_dict_args(self):
        msg = {'tool_calls': [{'function': {'name': 'get', 'arguments': {'a': 1}}}]}
        self.assertEqual(tool_calls(msg), [('get', {'a': 1})])

    def test_tool_calls_str_args(self):
        msg = {'tool_calls': [{'function': {'name': 'get', 'arguments': '{"a": 2}'}}]}
        self.assertEqual(tool_calls(msg), [('get', {'a': 2})])

    def test_tool_calls_invalid_str_args(self):
        msg = {'tool_calls': [{'function': {'name': 'get', 'arguments': 'not json'}}]}
        self.assertEqual(tool_calls(msg), [('get', {})])

    def test_tool_calls_missing_name_ignored(self):
        msg = {'tool_calls': [{'function': {'arguments': {}}}, {'function': {'name': 'ok', 'arguments': {}}}]}
        self.assertEqual(tool_calls(msg), [('ok', {})])
        self.assertEqual(tool_calls({}), [])


if __name__ == '__main__':
    unittest.main()

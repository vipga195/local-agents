"""Minimal Ollama chat client with tool calling."""

import json
import os
import urllib.error
import urllib.request


class BrainError(Exception):
    """Base exception for Brain errors."""
    pass


class OllamaBrain:
    """Client for Ollama /api/chat."""

    def __init__(self, host=None, model=None, timeout=300):
        self.host = host or os.environ.get('OLLAMA_HOST') or 'http://127.0.0.1:11434'
        if '://' not in self.host:
            self.host = 'http://' + self.host
        self.host = self.host.rstrip('/')
        self.model = model or os.environ.get('AGENT_MODEL') or 'qwen2.5:7b'
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a non-streaming chat request and return the reply message."""
        payload = {'model': self.model, 'messages': messages, 'stream': False, 'options': {'temperature': 0.2}}
        if tools:
            payload['tools'] = tools
        req = urllib.request.Request(
            self.host + '/api/chat',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode('utf-8')
        except urllib.error.HTTPError as e:  # Must precede URLError (subclass)
            body = e.read().decode('utf-8', errors='replace')
            raise BrainError(f'HTTP {e.code}: {body[:200]}') from e
        except urllib.error.URLError as e:
            raise BrainError(f'connection failed: {e.reason}') from e
        except TimeoutError as e:
            raise BrainError(f'timeout after {self.timeout}s') from e
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise BrainError('invalid JSON from Ollama') from e
        message = data.get('message') if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise BrainError('reply has no valid "message" field')
        return message


def _post_json(url, headers, payload, timeout, secret, label):
    """POST JSON and return the parsed reply dict; raise BrainError on any failure."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:  # Must precede URLError (subclass)
        text = e.read().decode('utf-8', errors='replace')
        if secret:
            text = text.replace(secret, '***')
        raise BrainError(f'HTTP {e.code}: {text[:200]}') from e
    except urllib.error.URLError as e:
        raise BrainError(f'connection failed: {e.reason}') from e
    except TimeoutError as e:
        raise BrainError(f'timeout after {timeout}s') from e
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise BrainError(f'invalid JSON from {label}') from e
    if not isinstance(data, dict):
        raise BrainError(f'unexpected reply from {label}')
    return data


def _call_args(call):
    """Return (name, args dict) of an internal tool call."""
    fn = call.get('function') or {}
    args = fn.get('arguments') or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return fn.get('name') or '', args if isinstance(args, dict) else {}


class OpenAIBrain:
    """Client for OpenAI-compatible /chat/completions."""

    def __init__(self, host=None, model=None, api_key=None, timeout=300):
        self.host = (host or 'https://api.openai.com/v1').rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _convert(self, messages):
        out = []
        pending = []  # ids of the last assistant tool calls not yet answered
        counter = 0
        for m in messages:
            role = m.get('role')
            if role == 'assistant':
                item = {'role': 'assistant', 'content': m.get('content') or ''}
                pending = []
                calls = []
                for c in m.get('tool_calls') or []:
                    name, args = _call_args(c)
                    cid = f'call_{counter}'
                    counter += 1
                    pending.append(cid)
                    calls.append({'id': cid, 'type': 'function',
                                  'function': {'name': name, 'arguments': json.dumps(args)}})
                if calls:
                    item['tool_calls'] = calls
                out.append(item)
            elif role == 'tool':
                if pending:
                    cid = pending.pop(0)
                else:
                    cid = f'call_{counter}'
                    counter += 1
                out.append({'role': 'tool', 'tool_call_id': cid, 'content': m.get('content') or ''})
            else:
                out.append({'role': role, 'content': m.get('content') or ''})
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a non-streaming chat request and return the reply in internal format."""
        payload = {'model': self.model, 'messages': self._convert(messages), 'temperature': 0.2}
        if tools:
            payload['tools'] = tools
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        data = _post_json(self.host + '/chat/completions', headers, payload,
                          self.timeout, self.api_key, 'OpenAI')
        choices = data.get('choices')
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get('message') if isinstance(first, dict) else None
        if not isinstance(message, dict):
            raise BrainError('reply has no valid "choices[0].message" field')
        reply = {'role': 'assistant', 'content': message.get('content') or ''}
        calls = []
        for c in message.get('tool_calls') or []:
            if not isinstance(c, dict):
                continue
            name, args = _call_args(c)
            if name:
                calls.append({'function': {'name': name, 'arguments': args}})
        if calls:
            reply['tool_calls'] = calls
        return reply


class AnthropicBrain:
    """Client for the Anthropic Messages API."""

    def __init__(self, host=None, model=None, api_key=None, timeout=300, max_tokens=4096):
        self.host = (host or 'https://api.anthropic.com').rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _convert(self, messages):
        system = []
        out = []
        pending = []  # tool_use ids of the last assistant turn not yet answered
        counter = 0
        for m in messages:
            role = m.get('role')
            content = m.get('content') or ''
            if role == 'system':
                system.append(content)
            elif role == 'assistant':
                blocks = [{'type': 'text', 'text': content}] if content else []
                pending = []
                for c in m.get('tool_calls') or []:
                    name, args = _call_args(c)
                    cid = f'call_{counter}'
                    counter += 1
                    pending.append(cid)
                    blocks.append({'type': 'tool_use', 'id': cid, 'name': name, 'input': args})
                out.append({'role': 'assistant', 'content': blocks or [{'type': 'text', 'text': ''}]})
            elif role == 'tool':
                if pending:
                    cid = pending.pop(0)
                else:
                    cid = f'call_{counter}'
                    counter += 1
                block = {'type': 'tool_result', 'tool_use_id': cid, 'content': content}
                prev = out[-1] if out else None
                if prev and prev['role'] == 'user' and isinstance(prev['content'], list):
                    prev['content'].append(block)
                else:
                    out.append({'role': 'user', 'content': [block]})
            else:
                out.append({'role': 'user', 'content': content})
        return '\n\n'.join(system), out

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a non-streaming request and return the reply in internal format."""
        system, converted = self._convert(messages)
        payload = {'model': self.model, 'max_tokens': self.max_tokens,
                   'messages': converted, 'temperature': 0.2}
        if system:
            payload['system'] = system
        if tools:
            payload['tools'] = [
                {'name': t['function']['name'],
                 'description': t['function'].get('description', ''),
                 'input_schema': t['function'].get('parameters') or {'type': 'object', 'properties': {}}}
                for t in tools]
        headers = {'Content-Type': 'application/json', 'x-api-key': str(self.api_key),
                   'anthropic-version': '2023-06-01'}
        data = _post_json(self.host + '/v1/messages', headers, payload,
                          self.timeout, self.api_key, 'Anthropic')
        blocks = data.get('content')
        if not isinstance(blocks, list):
            raise BrainError('reply has no valid "content" field')
        texts = []
        calls = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get('type') == 'text':
                texts.append(b.get('text') or '')
            elif b.get('type') == 'tool_use' and b.get('name'):
                args = b.get('input')
                calls.append({'function': {'name': b['name'], 'arguments': args if isinstance(args, dict) else {}}})
        reply = {'role': 'assistant', 'content': ''.join(texts)}
        if calls:
            reply['tool_calls'] = calls
        return reply


def tool_calls(message: dict) -> list[tuple[str, dict]]:
    """Extract (name, arguments) pairs from a reply message."""
    result = []
    for entry in message.get('tool_calls') or []:
        fn = entry.get('function') or {}
        name = fn.get('name')
        if not name:
            continue
        args = fn.get('arguments') or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
        result.append((name, args))
    return result

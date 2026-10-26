"""Minimal MCP stdio client (newline-delimited JSON-RPC 2.0), stdlib only."""
import json
import os
import queue
import subprocess
import threading

_EOF = object()  # sentinel delivered to waiters when the server exits


class McpError(Exception):
    pass


class McpClient:
    def __init__(self, cmd, env=None, startup_timeout=60):
        self.cmd = cmd
        self.startup_timeout = startup_timeout
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._q_lock = threading.Lock()
        self._queues = {}
        self._closed = False
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {})},
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _queue_for(self, rid):
        # Create on demand so an early response is not lost.
        with self._q_lock:
            q = self._queues.get(rid)
            if q is None:
                q = self._queues[rid] = queue.Queue()
                if self._closed:
                    q.put(_EOF)
            return q

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict) and "id" in msg:
                    self._queue_for(msg["id"]).put(msg)
        except (ValueError, OSError):
            pass
        finally:
            with self._q_lock:
                self._closed = True
                for q in self._queues.values():
                    q.put(_EOF)

    def _send(self, obj):
        try:
            with self._write_lock:
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as e:
            raise McpError(f"server exited ({e})")

    def start(self):
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "local-agent", "version": "0.1"},
            },
            timeout=self.startup_timeout,
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exc):
        self.close()

    def request(self, method, params=None, timeout=60):
        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        q = self._queue_for(rid)
        try:
            msg = {"jsonrpc": "2.0", "id": rid, "method": method}
            if params is not None:
                msg["params"] = params
            self._send(msg)
            try:
                resp = q.get(timeout=timeout)
            except queue.Empty:
                raise McpError(f"timeout after {timeout}s calling {method}")
        finally:
            with self._q_lock:
                self._queues.pop(rid, None)
        if resp is _EOF:
            raise McpError("server exited")
        if "error" in resp:
            err = resp["error"]
            raise McpError(f"{err.get('code')}: {err.get('message')}"
                           if isinstance(err, dict) else str(err))
        return resp.get("result")

    def list_tools(self):
        return self.request("tools/list")["tools"]

    def call_tool(self, name, arguments=None, timeout=3600):
        result = self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )
        text = "\n".join(
            c.get("text", "")
            for c in result.get("content", [])
            if c.get("type") == "text"
        )
        if result.get("isError"):
            raise McpError(text)
        return text

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        # Let the reader hit EOF, then release the pipe.
        self._reader.join(timeout=2)
        try:
            self.proc.stdout.close()
        except (OSError, ValueError):
            pass

import os
import sys
import threading
import time
import unittest

from mcp_client import McpClient, McpError

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "fake_mcp_server.py")


def make_client(test):
    c = McpClient([sys.executable, SERVER], startup_timeout=10).start()
    test.addCleanup(c.close)
    return c


class McpClientTest(unittest.TestCase):
    def test_handshake_and_list_tools(self):
        c = make_client(self)
        names = [t["name"] for t in c.list_tools()]
        self.assertEqual(names, ["echo", "fail", "slow"])

    def test_echo(self):
        c = make_client(self)
        self.assertEqual(c.call_tool("echo", {"text": "hi"}, timeout=5), "hi")

    def test_is_error(self):
        c = make_client(self)
        with self.assertRaises(McpError) as cm:
            c.call_tool("fail", timeout=5)
        self.assertIn("boom", str(cm.exception))

    def test_unknown_tool(self):
        c = make_client(self)
        with self.assertRaises(McpError):
            c.call_tool("nope", timeout=5)

    def test_timeout(self):
        c = make_client(self)
        with self.assertRaises(McpError) as cm:
            c.call_tool("slow", {"text": "x"}, timeout=1)
        self.assertIn("timeout", str(cm.exception))

    def test_concurrent_calls(self):
        c = make_client(self)
        out = {}

        def slow():
            try:
                out["slow"] = c.call_tool("slow", {"text": "s"}, timeout=10)
            except Exception as e:
                out["slow"] = e

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.3)
        # The fake server is single-threaded, so echo is answered after slow;
        # both responses must still reach their own callers.
        out["echo"] = c.call_tool("echo", {"text": "e"}, timeout=10)
        t.join(10)
        self.assertEqual(out["echo"], "e")
        self.assertEqual(out["slow"], "s")

    def test_server_killed(self):
        c = make_client(self)
        c.proc.kill()
        start = time.time()
        with self.assertRaises(McpError):
            c.call_tool("echo", {"text": "x"}, timeout=10)
        self.assertLess(time.time() - start, 5)

    def test_server_killed_while_waiting(self):
        c = make_client(self)
        out = {}

        def wait():
            try:
                c.call_tool("slow", {"text": "x"}, timeout=30)
            except McpError as e:
                out["err"] = e

        t = threading.Thread(target=wait)
        t.start()
        time.sleep(0.5)
        start = time.time()
        c.proc.kill()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertLess(time.time() - start, 5)
        self.assertIn("err", out)

    def test_close_twice(self):
        c = make_client(self)
        c.close()
        c.close()

    def test_context_manager(self):
        with McpClient([sys.executable, SERVER], startup_timeout=10) as c:
            self.assertEqual(c.call_tool("echo", {"text": "cm"}, timeout=5),
                             "cm")
        self.assertIsNotNone(c.proc.poll())


if __name__ == "__main__":
    unittest.main()

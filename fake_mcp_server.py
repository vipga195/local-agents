"""Tiny stdlib MCP stdio server used only by tests."""
import json
import sys
import time

TOOLS = [
    {"name": "echo", "description": "Echo text",
     "inputSchema": {"type": "object",
                     "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "fail", "description": "Always fails",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "slow", "description": "Echo after 3s",
     "inputSchema": {"type": "object",
                     "properties": {"text": {"type": "string"}}}},
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def reply(rid, result=None, error=None):
    # Noise first, to prove the client ignores it.
    sys.stdout.write("not json\n")
    sys.stdout.flush()
    send({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
    msg = {"jsonrpc": "2.0", "id": rid}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    send(msg)


def text(t, is_error=False):
    r = {"content": [{"type": "text", "text": t}]}
    if is_error:
        r["isError"] = True
    return r


def main():
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if "id" not in req:
            continue
        rid, method = req["id"], req.get("method")
        params = req.get("params") or {}
        if method == "initialize":
            reply(rid, {"protocolVersion": "2024-11-05", "capabilities": {},
                        "serverInfo": {"name": "fake", "version": "0"}})
        elif method == "tools/list":
            reply(rid, {"tools": TOOLS})
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                reply(rid, text(args.get("text", "")))
            elif name == "fail":
                reply(rid, text("boom", True))
            elif name == "slow":
                time.sleep(3)
                reply(rid, text(args.get("text", "")))
            else:
                reply(rid, error={"code": -32602, "message": "unknown tool"})
        else:
            reply(rid, error={"code": -32601, "message": "method not found"})


if __name__ == "__main__":
    main()

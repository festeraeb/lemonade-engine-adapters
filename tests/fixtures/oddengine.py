#!/usr/bin/env python3
"""A tiny OpenAI-compatible server with deliberately unusual flag names, for tests only.

It is not an inference engine: it answers every request with a fixed sentence. The flag names
(--bind-port, --weights, --window, --max-new) are chosen so that the deterministic scorer
cannot tell which one is the context size.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser(prog="oddengine")
ap.add_argument("--bind-port", type=int, metavar="PORT", required=True, help="TCP port to serve on")
ap.add_argument("--weights", metavar="FILE", required=True, help="model weights file")
ap.add_argument("--window", type=int, metavar="N", default=512, help="how many tokens the model can see at once")
ap.add_argument("--max-new", type=int, metavar="N", default=64, help="most tokens one answer may have")
a = ap.parse_args()


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.send(200, {"object": "list", "data": [{"id": "odd", "object": "model", "owned_by": "oddengine"}]})
        else:
            self.send(404, {"error": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/v1/chat/completions":
            if req.get("stream"):
                chunk = {"choices": [{"delta": {"content": "hello there"}, "index": 0}]}
                self.send(200, f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(), "text/event-stream")
            else:
                self.send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello there"}}]})
        elif self.path == "/v1/completions":
            self.send(200, {"choices": [{"index": 0, "text": "hello"}]})
        else:
            self.send(404, {"error": "no"})


HTTPServer(("127.0.0.1", a.bind_port), H).serve_forever()

"""Serve an output folder (report.html + page images) behind HTTP basic auth, on localhost only.

Put it on the internet with a Cloudflare quick tunnel:
    cloudflared tunnel --config /dev/null --no-autoupdate --url http://127.0.0.1:<port>
(`--config` stops cloudflared from picking up a named tunnel's ~/.cloudflared/config.yml.)
"""
from __future__ import annotations

import base64
import functools
import hmac
import http.server


def serve(root: str, port: int, user: str, password: str) -> None:
    token = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get("Authorization") or "", token):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="OCR report"')
                self.end_headers()
                return
            if self.path in ("/", "/index.html"):
                self.path = "/report.html"
            super().do_GET()

        def do_HEAD(self):
            self.do_GET()

        def list_directory(self, path):
            self.send_error(404)

        def log_message(self, *a):
            pass

    print(f"serving {root} on http://127.0.0.1:{port} (user {user})", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=root)).serve_forever()

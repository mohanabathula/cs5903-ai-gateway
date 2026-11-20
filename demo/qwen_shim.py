import http.server, json, urllib.request

UP = "http://127.0.0.1:9100"

class H(http.server.BaseHTTPRequestHandler):
    def _proxy(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(UP + self.path, data=body, method=self.command)
        for k in ("Content-Type", "Authorization"):
            if self.headers.get(k):
                req.add_header(k, self.headers[k])
        try:
            r = urllib.request.urlopen(req, timeout=300)
            code, data = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, data = e.code, e.read()
        try:
            j = json.loads(data)
            if isinstance(j, dict) and "metrics" in j:
                j.pop("metrics")
                data = json.dumps(j).encode()
        except Exception:
            pass
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    do_GET = do_POST = _proxy

http.server.ThreadingHTTPServer(("172.17.0.1", 9000), H).serve_forever()

"""Check an installed copy of the package serves everything. Run it outside the source tree."""
from __future__ import annotations

import json
import sys
import threading
from urllib.request import urlopen

import myco_fusion
from myco_fusion.server import make_server


def main():
    if "site-packages" not in myco_fusion.__file__:
        print(f"Warning: testing {myco_fusion.__file__}, not an installed copy")
    server = make_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    failures = 0
    for route in ("/", "/app.js", "/findings.js", "/styles.css", "/api/status", "/api/findings", "/api/example", "/api/calibration-example", "/api/repeatability-example"):
        try:
            with urlopen(base + route, timeout=30) as response:
                body = response.read()
            ok = response.status == 200 and len(body) > 10
            print(f"{'ok  ' if ok else 'FAIL'} {route} ({len(body)} bytes)")
            failures += not ok
        except Exception as exc:
            print(f"FAIL {route}: {exc}")
            failures += 1
    with urlopen(base + "/api/findings", timeout=30) as response:
        if len(json.load(response).get("findings", [])) < 5:
            print("FAIL findings look empty")
            failures += 1
    server.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

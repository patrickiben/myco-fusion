"""Loopback-only research UI. No uploads are saved and no cloud services are used."""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .manifest import build_manifest

APP = Path(__file__).resolve().parent / "app"
MAX_BODY = 2_000_000
STATIC = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/app.js": "app.js", "/findings.js": "findings.js"}


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _get_api(route):
    """Read-only data routes, loaded on demand."""
    if route == "/api/status":
        return {"app": "myco-fusion", "version": __version__, "mode": "local_research"}
    if route == "/api/findings":
        from .data import findings
        return findings()
    if route == "/api/calibration-example":
        from .calibration import calibration_example
        return calibration_example()
    if route == "/api/example":
        from .examples import validation_csv
        return {"csv": validation_csv()}
    if route == "/api/repeatability-example":
        from .examples import repeatability_csv
        return {"csv": repeatability_csv()}
    return None


def _post_api(route, data):
    if route == "/api/bench":
        from .bench import run_bench
        return run_bench(seed=data.get("seed", 7), noise=data.get("noise"), source=data.get("source", "measured"))
    if route == "/api/validate":
        from .validation import evaluate_csv
        if not isinstance(data.get("csv"), str):
            raise ValueError("csv must contain the paired-spectrum CSV text")
        return evaluate_csv(data["csv"], uncertainty=data.get("uncertainty", "auto"))
    if route == "/api/calibrate":
        from .calibration import fit_calibration
        return fit_calibration(data.get("dataset"))
    if route == "/api/predict":
        from .calibration import predict_calibration
        return predict_calibration(data.get("model"), data.get("capture"))
    if route == "/api/repeatability":
        from .repeatability import analyze_repeatability
        if not isinstance(data.get("csv"), str):
            raise ValueError("csv must contain the photo-table CSV text")
        return analyze_repeatability(data["csv"])
    if route == "/api/manifest":
        return build_manifest(data)
    return None


POST_ROUTES = {"/api/bench", "/api/validate", "/api/manifest", "/api/calibrate", "/api/predict", "/api/repeatability"}


class Handler(BaseHTTPRequestHandler):
    server_version = "MycoFusion/" + __version__

    def log_message(self, fmt, *args):
        # Do not log uploaded records or specimen identifiers.
        pass

    def respond(self, status, data, content_type="application/json; charset=utf-8"):
        payload = json.dumps(data, allow_nan=False).encode() if isinstance(data, dict) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self.valid_host():
            return self.respond(403, {"error": "Use the localhost address printed by the app."})
        route = urlsplit(self.path).path
        if route.startswith("/api/"):
            try:
                result = _get_api(route)
            except ImportError:
                return self.respond(503, {"error": "NumPy is not installed. Run: pip install ."})
            except OSError:
                return self.respond(503, {"error": "A bundled data file is missing. Reinstall the package."})
            if result is None:
                return self.respond(404, {"error": "Not found"})
            return self.respond(200, result)
        if route not in STATIC:
            return self.respond(404, {"error": "Not found"})
        path = APP / STATIC[route]
        if not path.is_file():
            return self.respond(503, {"error": "The interface files are missing. Reinstall the package."})
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.respond(200, path.read_bytes(), mime + "; charset=utf-8")

    def do_POST(self):
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if not self.valid_host() or (origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}):
            return self.respond(403, {"error": "Requests must come from this local app."})
        route = urlsplit(self.path).path
        if route not in POST_ROUTES:
            return self.respond(404, {"error": "Unknown API route"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_BODY:
                return self.respond(413, {"error": "Supply a JSON request smaller than 2 MB."})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.respond(415, {"error": "Content-Type must be application/json"})
            data = json.loads(self.rfile.read(size), parse_constant=reject_constant)
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object")
            self.respond(200, _post_api(route, data))
        except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
            self.respond(400, {"error": str(exc)})
        except ImportError:
            self.respond(503, {"error": "NumPy is not installed. Run: pip install ."})
        except Exception:
            self.respond(500, {"error": "Unexpected computation error. Check the input and local test suite."})


def make_server(port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port=8765, open_browser=False):
    import webbrowser
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("Myco-fusion needs NumPy. Install the package first: pip install .") from None
    with make_server(port) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Myco-fusion {__version__} → {url}\nLocal research workbench. Ctrl+C to stop.", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

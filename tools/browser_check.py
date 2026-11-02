"""Open every tab in a real browser and fail if anything stays empty or breaks.

Needs Playwright (pip install playwright, then playwright install chromium). Run:
    python tools/browser_check.py [--screenshots DIR]
"""
from __future__ import annotations

import argparse
import struct
import sys
import tempfile
import threading
import zlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

from myco_fusion.server import make_server  # noqa: E402

AUDIT = r"""
() => {
  const luminance = ([r, g, b]) => { const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const parse = (color) => { const m = color.match(/[\d.]+/g).map(Number); return m.length >= 4 ? { rgb: m.slice(0, 3), a: m[3] } : { rgb: m.slice(0, 3), a: 1 }; };
  const background = (node) => { for (let n = node; n && n.nodeType === 1; n = n.parentElement) { const c = parse(getComputedStyle(n).backgroundColor); if (c.a >= 0.99) return c.rgb; } return [255, 255, 255]; };
  const contrast = (a, b) => { const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const problems = { lowContrast: [], tiny: [], overflow: [] };
  document.querySelectorAll('details').forEach((d) => { d.open = true; });
  document.querySelectorAll('body *').forEach((node) => {
    if (node.closest('svg') && !['text'].includes(node.tagName.toLowerCase())) return;
    const own = [...node.childNodes].some((c) => c.nodeType === 3 && c.textContent.trim());
    if (!own || node.getClientRects().length === 0 || node.closest('[hidden]')) return;
    const style = getComputedStyle(node);
    if (style.visibility === 'hidden' || Number(style.opacity) === 0) return;
    const size = parseFloat(style.fontSize), text = node.textContent.trim().slice(0, 50);
    if (size < 11) problems.tiny.push([size, text]);
    const svgText = node.tagName.toLowerCase() === 'text';
    const color = parse(svgText ? style.fill : style.color).rgb;
    const ratio = contrast(color, background(svgText ? node.closest('svg') : node));
    if (ratio < (size >= 24 ? 3 : 4.5)) problems.lowContrast.push([Number(ratio.toFixed(2)), size, text]);
    if (!svgText && node.scrollWidth > node.clientWidth + 2 && style.display !== 'inline' && style.overflowX === 'visible') problems.overflow.push(text);
  });
  document.querySelectorAll('.panel, .notice, .scenario-card, .stat-tile').forEach((panel) => {
    if (panel.getClientRects().length === 0) return;
    const box = panel.getBoundingClientRect();
    panel.querySelectorAll('h1, h2, h3, p, strong, span, td, button, label').forEach((child) => {
      if (child.getClientRects().length === 0 || child.closest('.table-scroll')) return;
      const inner = child.getBoundingClientRect();
      if (inner.right > box.right + 2 || inner.left < box.left - 2) problems.overflow.push(`past its panel: ${child.textContent.trim().slice(0, 50)}`);
    });
  });
  problems.pageOverflow = document.documentElement.scrollWidth - innerWidth;
  return problems;
}
"""


def solid_png(path, rgb, size=64):
    """A tiny solid-color PNG, written without any imaging library."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def check_photo_tool(page, require, name):
    """Measure known colors through the real file input, canvas and table."""
    with tempfile.TemporaryDirectory() as folder:
        files = []
        for object_name, rgb in (("mug", (200, 100, 50)), ("cup", (60, 180, 90)), ("box", (255, 255, 255))):
            for session in (1, 2):
                for repeat in (1, 2):
                    path = Path(folder) / f"{object_name}_s{session}_flash_r{repeat}.png"
                    solid_png(path, rgb)
                    files.append(str(path))
        bad = Path(folder) / "IMG_1234.png"
        solid_png(bad, (10, 10, 10))
        files.append(str(bad))
        page.locator("#phone-input").fill("")
        page.set_input_files("#roi-files", files)
        page.wait_for_selector("#roi-stage:not([hidden])")
        page.locator("#roi-canvas").scroll_into_view_if_needed()
        box = page.locator("#roi-canvas").bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.25)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.75, box["y"] + box["height"] * 0.75, steps=4)
        page.mouse.up()
        page.locator("#roi-measure").click()
        page.locator("#roi-status", has_text="Added 12 rows").wait_for(timeout=30000)
        status = page.locator("#roi-status").inner_text()
        require("Skipped 1" in status and "IMG_1234" in status, f"{name}: badly named photo is listed")
        require("clipped" in status, f"{name}: clipped photos are flagged")
        table = page.input_value("#phone-input")
        require(len(table.strip().splitlines()) == 13, f"{name}: one row per photo")
        mug = next(line for line in table.splitlines() if line.startswith("mug,s1,flash,1"))
        red = float(mug.split(",")[4])
        require(abs(red - 0.5776) < 0.002, f"{name}: linearized value is right, got {red}")
        require(page.locator("#phone-file-name").inner_text() == "From your photos", f"{name}: table source is relabeled")
        page.locator("#analyze-phone").click()
        page.locator("#phone-message", has_text="Analysis complete").wait_for(timeout=30000)
        require(page.locator("#phone-table tr").count() == 3, f"{name}: analysis of measured photos")


def wait_for_value(page, selector, timeout_ms=20000):
    """The app's security policy blocks eval-based waits, so poll the field instead."""
    for _ in range(timeout_ms // 100):
        if len(page.input_value(selector)) > 100:
            return
        page.wait_for_timeout(100)
    raise TimeoutError(f"{selector} stayed empty")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    server = make_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    failures, notes = [], []

    def require(condition, label):
        if not condition:
            failures.append(label)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    site = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "site")))
    threading.Thread(target=site.serve_forever, daemon=True).start()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, viewport in (("site-desktop", {"width": 1280, "height": 900}), ("site-phone", {"width": 390, "height": 844})):
            page = browser.new_page(viewport=viewport)
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.goto(f"http://127.0.0.1:{site.server_port}/")
            page.wait_for_selector(".finding svg")
            require(page.locator(".finding").count() == 9, f"{name}: nine findings")
            require(page.locator(".finding-chart svg").count() == 8, f"{name}: eight charts")
            require("safe to eat" in page.locator(".notice.safety").inner_text(), f"{name}: safety notice")
            if args.screenshots:
                args.screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshots / f"{name}.png"), full_page=True)
            report = page.evaluate(AUDIT)
            for kind in ("lowContrast", "tiny", "overflow"):
                if report[kind]:
                    notes.append(f"{name} {kind}: {report[kind][:6]} ({len(report[kind])} total)")
            if report["pageOverflow"] > 1:
                failures.append(f"{name}: page scrolls sideways by {report['pageOverflow']} px")
            require(not errors, f"{name}: console errors {errors[:3]}")
            page.close()
        for name, viewport in (("desktop", {"width": 1280, "height": 900}), ("phone", {"width": 390, "height": 844})):
            page = browser.new_page(viewport=viewport)
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.goto(f"{url}/#findings")
            page.wait_for_selector(".finding svg")
            require(page.locator(".finding").count() == 9, f"{name}: nine findings")
            require(page.locator(".finding-chart svg").count() == 8, f"{name}: eight charts")
            require("safe to eat" in page.locator(".notice.safety").inner_text(), f"{name}: safety notice")

            page.goto(f"{url}/#bench")
            page.wait_for_selector(".scenario-card")
            page.wait_for_selector("#spectrum-chart svg")
            require(page.locator(".scenario-card").count() == 4, f"{name}: four bench scenarios")
            page.locator(".scenario-card").nth(3).click()
            require(page.locator("#rank-number").inner_text().strip() not in ("", "—", "3"), f"{name}: screen scenario adds measurements")
            require(page.locator("#metamer-chart svg").count() == 1, f"{name}: metamer chart")
            require("plum" in page.locator("#metamer-a").inner_text().lower() + page.locator("#metamer-b").inner_text().lower(), f"{name}: real metamer pair")
            page.locator("#example-worst").click()
            page.select_option("#bench-source", "idealized")
            page.locator("#bench-badge", has_text="BEST CASE").wait_for()
            page.select_option("#bench-source", "measured")
            page.locator("#bench-badge", has_text="SIMULATED").wait_for()

            page.goto(f"{url}/#phone")
            page.locator("#load-phone-demo").click()
            wait_for_value(page, "#phone-input")
            page.locator("#analyze-phone").click()
            page.wait_for_selector("#phone-results:not([hidden])", timeout=60000)
            require(page.locator("#phone-table tr").count() == 12, f"{name}: repeatability table")
            require(page.locator(".phone-rank-row").count() == 5, f"{name}: independent-measurement rows")
            require("Crossed design" in page.locator("#phone-design-note").inner_text(), f"{name}: crossed design noted")
            page.locator("#open-light-panel").click()
            require(page.locator("#light-panel").is_visible(), f"{name}: light panel opens")
            page.keyboard.press("ArrowRight")
            require(page.locator("#light-name").inner_text() == "Green", f"{name}: light panel advances")
            page.keyboard.press("Escape")
            page.wait_for_selector("#light-panel", state="hidden")
            page.wait_for_timeout(500)  # Leaving full screen takes a moment.
            require(page.evaluate("() => document.fullscreenElement === null"), f"{name}: full screen is closed")

            check_photo_tool(page, require, name)

            page.goto(f"{url}/#calibration")
            page.locator("#load-calibration-demo").click()
            wait_for_value(page, "#calibration-input")
            page.locator("#fit-calibration").click()
            page.wait_for_selector("#calibration-results:not([hidden])", timeout=60000)
            require(page.locator("#calibration-regimes tr").count() == 2, f"{name}: calibration regimes")
            require("96.9% interval" in page.locator("#calibration-regimes").inner_text(), f"{name}: calibration interval level")
            require(page.locator("#calibration-chart svg").count() == 1, f"{name}: calibration chart")

            page.goto(f"{url}/#validate")
            page.locator("#load-demo").click()
            wait_for_value(page, "#csv-input")
            page.locator("#validate-csv").click()
            page.wait_for_selector("#validation-results:not([hidden])", timeout=60000)
            require(page.locator("#validation-table tr").count() == 2, f"{name}: validation regimes")
            require(" of 48" in page.locator("#validation-table").inner_text(), f"{name}: specimens within targets")

            page.goto(f"{url}/#capture")
            page.locator('input[name="specimen_id"]').fill("demo-1")
            page.locator('input[name="session_id"]').fill("s1")
            page.locator('input[name="device"]').fill("phone")
            page.locator('input[name="reference_id"]').fill("none yet")
            page.locator("#create-manifest").click()
            page.wait_for_selector("#manifest-result:not([hidden])")

            for view in ("findings", "bench", "phone", "calibration", "validate", "capture"):
                page.goto(f"{url}/#{view}")
                page.wait_for_timeout(350)
                if view == "phone":
                    page.locator("#load-phone-demo").click()
                    wait_for_value(page, "#phone-input")
                    page.locator("#analyze-phone").click()
                    page.wait_for_selector("#phone-results:not([hidden])", timeout=60000)
                if view == "calibration":
                    page.locator("#load-calibration-demo").click()
                    wait_for_value(page, "#calibration-input")
                    page.locator("#fit-calibration").click()
                    page.wait_for_selector("#calibration-results:not([hidden])", timeout=60000)
                if view == "validate":
                    page.locator("#load-demo").click()
                    wait_for_value(page, "#csv-input")
                    page.locator("#validate-csv").click()
                    page.wait_for_selector("#validation-results:not([hidden])", timeout=60000)
                if args.screenshots:
                    args.screenshots.mkdir(parents=True, exist_ok=True)
                    page.wait_for_timeout(200)
                    page.screenshot(path=str(args.screenshots / f"{name}-{view}.png"), full_page=True)
                report = page.evaluate(AUDIT)
                for kind in ("lowContrast", "tiny", "overflow"):
                    if report[kind]:
                        notes.append(f"{name}/{view} {kind}: {report[kind][:6]} ({len(report[kind])} total)")
                if report["pageOverflow"] > 1:
                    failures.append(f"{name}/{view}: page scrolls sideways by {report['pageOverflow']} px")
            require(not errors, f"{name}: console errors {errors[:3]}")
            page.close()
        browser.close()
    server.shutdown()
    site.shutdown()
    for note in notes:
        print("AUDIT", note)
    for failure in failures:
        print("FAIL", failure)
    print(f"{len(failures)} failures, {len(notes)} audit notes")
    return 1 if failures or notes else 0


if __name__ == "__main__":
    sys.exit(main())

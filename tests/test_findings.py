import json
import re
import unittest
from pathlib import Path

from myco_fusion.data import findings
from study.build_report import build, check_claims

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".md", ".py", ".js", ".html", ".css", ".json", ".yml", ".toml", ".txt", ".cff", ".command"}
# Terms that must stay out of the public repository live in a local, gitignored file.
DENYLIST = ROOT / "private_data" / "public_denylist.txt"


def public_files():
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if parts[0] in {"private_data", ".git", ".venv", "build", "dist"} or "__pycache__" in parts or parts[:2] == ("data", "raw"):
            continue
        yield path


class FindingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = json.loads((ROOT / "results" / "study.json").read_text())
        cls.coverage = json.loads((ROOT / "results" / "coverage.json").read_text())

    def test_results_still_support_the_written_claims(self):
        check_claims(self.study, self.coverage)

    def test_packaged_findings_match_the_results_files(self):
        self.assertEqual(findings(), json.loads(json.dumps(build(self.study, self.coverage), ensure_ascii=False)))

    def test_site_is_in_sync_with_the_app(self):
        self.assertEqual(json.loads((ROOT / "site" / "findings.json").read_text()), findings())
        for name in ("findings.js", "styles.css"):
            self.assertEqual((ROOT / "site" / name).read_text(), (ROOT / "myco_fusion" / "app" / name).read_text(), name)

    def test_readme_numbers_match_the_findings(self):
        readme = (ROOT / "README.md").read_text()
        for tile in findings()["tiles"]:
            self.assertIn(f"| {tile['label']} | {tile['value']} |", readme)
        natural = self.study["q1"]["natural"]
        primary = self.study["q4"]["primary"]["natural"]
        arm = primary["flash_plus_screen_rgb"]["reduction_vs_flash"]
        minus = lambda value: f"{round(value) + 0:d}".replace("-", "−")
        for expected in (f"{natural['affine']['sam_median']:.2f}°", f"{natural['affine']['sam_worst']:.0f}°",
                         f"{minus(arm['percent'])}% (95% interval {minus(arm['low'])} to {minus(arm['high'])})",
                         f"{primary['flash']['mae_p90']:.3f} to {primary['flash_plus_screen_rgb']['mae_p90']:.3f}",
                         f"{100 * natural['sigma_camera']['share_within_targets']:.0f}%",
                         f"{self.study['q2']['domains']['Leaves']['ratio']:.1f} times"):
            self.assertIn(expected, readme)
        phones = self.study["q7"].values()
        similar = [100 * e["affine"]["share_within_targets"] for e in phones]
        chart = [100 * e["chart"]["share_within_targets"] for e in phones]
        self.assertIn(f"{min(similar):.0f}–{max(similar):.0f}% and {min(chart):.0f}–{max(chart):.0f}%", readme)

    def test_all_tests_table_reports_failures_too(self):
        outcomes = {row[-1] for row in findings()["all_tests"]["rows"]}
        self.assertIn("Not met", outcomes)
        self.assertIn("Met", outcomes)
        self.assertGreater(len(findings()["all_tests"]["rows"]), 50)

    def test_every_finding_has_a_statement_and_safety_comes_first(self):
        data = findings()
        self.assertIn("safe to eat", data["safety"])
        for finding in data["findings"]:
            self.assertTrue(finding["title"] and finding["body"])

    @unittest.skipUnless(DENYLIST.exists(), "local denylist not present")
    def test_no_private_terms_in_public_files(self):
        terms = [line.strip() for line in DENYLIST.read_text(encoding="utf-8").splitlines() if line.strip()]
        for path in public_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for term in terms:
                self.assertNotIn(term, text, f"a private term appears in {path}")

    def test_markdown_links_resolve(self):
        for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), ROOT / "study" / "PLAN.md"]:
            for target in re.findall(r"\]\(([^)#\s]+)\)", path.read_text(encoding="utf-8")):
                if not target.startswith(("http://", "https://")):
                    self.assertTrue((path.parent / target).exists(), f"{target} in {path}")


if __name__ == "__main__":
    unittest.main()

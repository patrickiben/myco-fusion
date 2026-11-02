import tomllib
import unittest
from pathlib import Path

import myco_fusion

ROOT = Path(__file__).resolve().parent.parent


class PackageTests(unittest.TestCase):
    def test_version_matches_project_metadata(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["version"], myco_fusion.__version__)

    def test_runtime_assets_are_inside_the_package(self):
        package = Path(myco_fusion.__file__).parent
        for name in ("app/index.html", "app/app.js", "app/styles.css", "data/natural_objects_27band.csv", "data/findings.json"):
            self.assertTrue((package / name).is_file(), name)

    def test_private_material_is_ignored(self):
        ignored = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("private_data/", ignored)
        self.assertIn("data/raw/", ignored)


if __name__ == "__main__":
    unittest.main()

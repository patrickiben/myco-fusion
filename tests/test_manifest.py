import hashlib
import json
import unittest
from myco_fusion.manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def test_draft_and_full_digest(self):
        result = build_manifest(dict(specimen_id="spec-1", session_id="s-1", device="Phone", regime="R1", reference_id="tile-1"))
        record = result["manifest"].copy()
        digest = record.pop("record_sha256")
        record.pop("digest_scope")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())
        self.assertEqual(record["measurement_kind"], "not_acquired")
        self.assertFalse(any(record["checks"].values()))

    def test_missing_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "specimen_id"):
            build_manifest({})

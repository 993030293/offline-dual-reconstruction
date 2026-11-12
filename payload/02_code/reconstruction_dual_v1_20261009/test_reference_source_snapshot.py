"""Check long-path-safe freezing of the offline target implementation."""

import json
from pathlib import Path
import tempfile
import unittest

from run_target_pipeline import (
    capture_reference_sources,
    file_sha256,
    reference_python_sources,
    verify_reference_sources,
)
from target_prepare import PACKAGE


class ReferenceSourceSnapshotTests(unittest.TestCase):
    def test_deep_reference_source_is_readable_and_frozen(self):
        relative = Path(
            "code/reference_sources/e26_c2_support/frontend/tools/camera_motion.py"
        )
        expected = PACKAGE / relative
        sources = dict(
            (source.relative_to(PACKAGE), source)
            for source, _ in reference_python_sources(PACKAGE)
        )
        self.assertIn(relative, sources)
        self.assertEqual(file_sha256(sources[relative]), file_sha256(expected))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            manifest = {
                "sources": capture_reference_sources(PACKAGE, target),
                "captured_before_execution": True,
            }
            verify_reference_sources(manifest)
            row = next(
                item for item in manifest["sources"]
                if Path(item["source"]) == expected
            )
            self.assertTrue(Path(row["snapshot"]).is_file())
            self.assertEqual(file_sha256(row["snapshot"]), row["sha256"])
            self.assertNotIn("__pycache__", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()

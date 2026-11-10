"""The installed legacy branch must resolve its shared Bayer decoder imports."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIO_TOOLS = ROOT / "code" / "reference_sources" / "c0_lio_fallback" / "tools"
SFM_TOOLS = ROOT / "code" / "reference_sources" / "e26_c2_support" / "frontend" / "tools"


class LegacyLioImportsTest(unittest.TestCase):
    def test_wrapper_includes_shared_bayer_module_path(self) -> None:
        wrapper = (ROOT / "RUN_LEGACY_PGSR_S00_S08.ps1").read_text(encoding="utf-8-sig")
        self.assertIn('$env:PYTHONPATH = "$lioTools;$sfmTools;$rosbagsVendorPath"', wrapper)

    def test_legacy_entrypoints_import_with_packaged_modules(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(LIO_TOOLS), str(SFM_TOOLS)))
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in ("endpoint_visual_constraint.py", "lio_local_map.py"):
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, "-B", str(LIO_TOOLS / name), "--help"],
                    env=environment, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--mcap", result.stdout)


if __name__ == "__main__":
    unittest.main()

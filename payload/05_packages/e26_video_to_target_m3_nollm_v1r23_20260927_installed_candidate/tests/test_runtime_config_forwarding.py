"""Keep the full runner's runtime configuration bound through M3."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigForwardingTests(unittest.TestCase):
    def test_full_runner_forwards_runtime_config_to_m3(self) -> None:
        full = (ROOT / "RUN_UNATTENDED.ps1").read_text(encoding="utf-8")
        m3 = (ROOT / "RUN_FROM_S07_M3.ps1").read_text(encoding="utf-8")
        self.assertIn("-RuntimeConfig $RuntimeConfig", full)
        self.assertIn("[string]$RuntimeConfig", m3)
        self.assertIn("$runtimeConfig = [System.IO.Path]::GetFullPath($RuntimeConfig)", m3)
        self.assertIn("Get-Content -LiteralPath $runtimeConfig", m3)


if __name__ == "__main__":
    unittest.main()

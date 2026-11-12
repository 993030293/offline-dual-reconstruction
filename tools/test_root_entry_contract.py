"""Exercise the package verifier through its user-facing default entry."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = ROOT / "VERIFY_PACKAGE.ps1"
RUNTIME_PYTHON = ROOT / "payload/05_envs/sfm_py311_v1/Scripts/python.exe"
RUNTIME_BINDING = ROOT / "config/runtime.binding.json"
LOCAL_BROWSER_BINDING = ROOT / "payload/02_code/reconstruction_dual_v1_20261009/OFFLINE_RUNTIME.local.json"
RESULT = ROOT / "PACKAGE_VERIFY_RESULT.json"


class RootEntryContractTests(unittest.TestCase):
    def test_verify_package_default_output_is_executable(self):
        powershell = shutil.which("powershell")
        if powershell is None:
            self.skipTest("Windows PowerShell is unavailable")
        if not RUNTIME_PYTHON.exists():
            self.skipTest("The machine-local runtime has not been bound")

        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(VERIFY_SCRIPT),
                "-RequireRuntime",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        payload = json.loads(RESULT.read_text(encoding="utf-8-sig"))
        self.assertEqual(payload["status"], "package_verified")
        self.assertTrue(payload["runtime_required"])
        self.assertEqual(payload["runtime_failures"], [])
        browser_checks = [
            item for item in payload["runtime_checks"]
            if item.get("kind") == "offline_browser_binding"
        ]
        self.assertEqual(len(browser_checks), 1)
        self.assertTrue(browser_checks[0]["target_matches"])

    def test_local_browser_binding_matches_frozen_declaration(self):
        if not RUNTIME_PYTHON.exists():
            self.skipTest("The machine-local runtime has not been bound")
        self.assertTrue(LOCAL_BROWSER_BINDING.is_file())
        declared = json.loads(RUNTIME_BINDING.read_text(encoding="utf-8-sig"))["browser_runtime"]
        local = json.loads(LOCAL_BROWSER_BINDING.read_text(encoding="utf-8-sig"))
        self.assertEqual(Path(local["node"]), Path(declared["node"]))
        self.assertEqual(
            Path(local["playwright_package"]),
            Path(declared["playwright_package"]),
        )
        self.assertFalse(declared["runtime_download_allowed"])
        self.assertFalse(local["runtime_download_allowed"])


if __name__ == "__main__":
    unittest.main()

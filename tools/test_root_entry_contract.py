"""Exercise the package verifier through its user-facing default entry."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = ROOT / "VERIFY_PACKAGE.ps1"
RUNTIME_PYTHON = ROOT / "payload/05_envs/sfm_py311_v1/Scripts/python.exe"
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


if __name__ == "__main__":
    unittest.main()

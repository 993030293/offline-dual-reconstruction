"""Check evidence-preserving retries from a downstream shared-input failure."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import psutil

from common import HERE
from manual_retry import archive_failed_delivery_stage, archive_failed_front


def write(path, value=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        path.write_text("fixture", encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def process(status):
    return {"status": status, "owned_process_identities": []}


class ManualRetryTests(unittest.TestCase):
    def test_shared_failure_archives_completed_front_and_invalidated_leafs(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        side = root / "case" / "optimized"
        write(root / "STATE.json", {
            "status": "failed_delivery",
            "current_stage": "shared",
            "stages": [{"stage": "front"}],
        })
        write(root / "stages/front/PROCESS.json", process("completed"))
        write(root / "stages/front/COMPLETED.json", {"status": "completed"})
        write(root / "stages/shared/PROCESS.json", process("failed_child"))
        write(root / "stages/shared/run.log")
        write(root / "shared/partial.txt")
        for label in ("metric", "roles", "export"):
            write(side / (label + "_process.json"), process("completed"))
            write(side / (label + ".log"))
            write(side / (label + "_wrapper.json"), {})
            write(side / label / "RESULT.json", {})
        write(side / "camera_backend/RESULT.json", {"status": "preserved"})

        archive = archive_failed_front(root, "case", "metric", "fixture repair")

        self.assertFalse((root / "stages/front").exists())
        self.assertFalse((root / "stages/shared").exists())
        self.assertFalse((root / "shared").exists())
        for label in ("metric", "roles", "export"):
            self.assertFalse((side / label).exists())
            self.assertFalse((side / (label + "_process.json")).exists())
        self.assertTrue((side / "camera_backend/RESULT.json").is_file())
        receipt = json.loads((archive / "RETRY.json").read_text())
        self.assertEqual(receipt["outer_failure_boundary"], "shared")
        self.assertEqual(receipt["invalidated_front_leafs"], ["metric", "roles", "export"])
        self.assertFalse(receipt["single_unassisted_cold_start_claim_allowed"])

    def test_target_input_failure_archives_partial_target_and_preserves_shared(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        write(root / "STATE.json", {
            "status": "failed_delivery",
            "current_stage": "target_inputs",
            "stages": [{"stage": "shared"}],
        })
        write(root / "stages/shared/COMPLETED.json", {"status": "completed"})
        write(root / "stages/target_inputs/PROCESS.json", process("failed_child"))
        write(root / "stages/target_inputs/run.log")
        write(root / "target/PARTIAL.json", {"status": "failed"})
        write(root / "shared/RESULT.json", {"status": "preserved"})

        archive = archive_failed_delivery_stage(
            root, "target_inputs", "fixture long-path repair"
        )

        self.assertFalse((root / "stages/target_inputs").exists())
        self.assertFalse((root / "target").exists())
        self.assertTrue((root / "stages/shared/COMPLETED.json").is_file())
        self.assertTrue((root / "shared/RESULT.json").is_file())
        receipt = json.loads((archive / "RETRY.json").read_text())
        self.assertEqual(receipt["outer_failure_boundary"], "target_inputs")
        self.assertFalse(receipt["single_unassisted_cold_start_claim_allowed"])

    def test_delivery_failure_archives_only_partial_delivery_outputs(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        write(root / "STATE.json", {
            "status": "failed_delivery",
            "current_stage": "delivery",
            "stages": [{"stage": "target_geometry"}],
        })
        write(root / "stages/target_geometry/COMPLETED.json", {"status": "completed"})
        write(root / "stages/delivery/PROCESS.json", process("failed_child"))
        write(root / "target/RESULT.json", {"status": "target_geometry_complete"})
        write(root / "target/delivery/gaussian_scene.ply")
        write(root / "viewer/partial.txt")

        archive = archive_failed_delivery_stage(
            root, "delivery", "fixture packaged-viewer repair"
        )

        self.assertFalse((root / "stages/delivery").exists())
        self.assertFalse((root / "target/delivery").exists())
        self.assertFalse((root / "viewer").exists())
        self.assertTrue((root / "target/RESULT.json").is_file())
        self.assertTrue((root / "stages/target_geometry/COMPLETED.json").is_file())
        receipt = json.loads((archive / "RETRY.json").read_text())
        self.assertEqual(receipt["outer_failure_boundary"], "delivery")
        self.assertFalse(receipt["single_unassisted_cold_start_claim_allowed"])

    def test_delivery_retry_stops_owned_preview_before_process_drain_check(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        viewer = root / "viewer"
        viewer.mkdir(parents=True)
        command = [sys.executable, "-c", "import time; time.sleep(60)",
                   str(HERE / "serve_delivery.py"), str(viewer)]
        preview = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
        self.addCleanup(lambda: preview.poll() is None and preview.kill())
        birth = psutil.Process(preview.pid).create_time()
        write(root / "STATE.json", {
            "status": "failed_delivery",
            "current_stage": "delivery",
            "stages": [{"stage": "target_geometry"}],
        })
        write(root / "stages/target_geometry/COMPLETED.json", {"status": "completed"})
        write(root / "stages/delivery/PROCESS.json", {
            "status": "failed_child",
            "owned_process_identities": [{"pid": preview.pid, "creation_time": birth}],
        })
        write(root / "PREVIEW_SERVER.json", {
            "pid": preview.pid,
            "birth": birth,
            "command": command,
            "url": "http://127.0.0.1:1/",
            "run_root": str(root),
        })

        archive = archive_failed_delivery_stage(
            root, "delivery", "fixture stop-owned-preview repair"
        )

        preview.wait(timeout=10)
        self.assertIsNotNone(preview.returncode)
        self.assertFalse((root / "PREVIEW_SERVER.json").exists())
        self.assertTrue((archive / "retained/PREVIEW_SERVER.json").is_file())


if __name__ == "__main__":
    unittest.main()

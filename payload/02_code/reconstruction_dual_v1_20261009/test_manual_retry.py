"""Check evidence-preserving retries from a downstream shared-input failure."""

import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

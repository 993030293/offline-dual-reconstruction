"""Exercise source ownership and complete-byte checks used by the paired audit."""

from pathlib import Path
import tempfile
import unittest

from validate_pair import audit_training_sources, data_identity, windows_to_wsl


class TrainingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.campaigns = [Path(self.temporary.name) / name for name in ("reference", "candidate")]
        self.configs = []
        self.targets = []
        for campaign in self.campaigns:
            target = campaign / "box1/optimized/training/source_snapshot/train_case.py"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"print('same complete program')\n")
            identity = data_identity(target)
            identity["path"] = windows_to_wsl(target)
            self.configs.append({"implementation": [identity]})
            self.targets.append(target)

    def test_distinct_private_paths_with_equal_complete_sources(self):
        self.assertNotEqual(self.configs[0]["implementation"], self.configs[1]["implementation"])
        self.assertTrue(audit_training_sources(self.campaigns, self.configs)["passed"])

    def test_changed_source_rejected_even_with_updated_declared_identity(self):
        target = self.targets[1]
        target.write_bytes(b"print('different complete program')\n")
        identity = data_identity(target)
        identity["path"] = windows_to_wsl(target)
        self.configs[1]["implementation"] = [identity]
        result = audit_training_sources(self.campaigns, self.configs)
        self.assertFalse(result["passed"])
        self.assertEqual(result["different_complete_sources"], ["train_case.py"])

    def test_wrong_execution_path_rejected(self):
        self.configs[1]["implementation"][0]["path"] = self.configs[0]["implementation"][0]["path"]
        self.assertFalse(audit_training_sources(self.campaigns, self.configs)["passed"])


if __name__ == "__main__":
    unittest.main()

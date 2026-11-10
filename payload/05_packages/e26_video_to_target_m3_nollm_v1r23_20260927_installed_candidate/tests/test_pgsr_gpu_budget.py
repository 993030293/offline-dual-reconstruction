"""Check PGSR GPU admission against both physical and monitored headroom."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "validated_fresh_runtime" / "train_pgsr_supervised.py"
SPEC = importlib.util.spec_from_file_location("train_pgsr_supervised", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PgsrGpuBudgetTest(unittest.TestCase):
    def test_current_c1_resource_range_is_admitted(self) -> None:
        budget = MODULE.assess_gpu_budget({"free_mib": 7268, "used_mib": 8783})
        self.assertEqual(budget["status"], "admitted")
        self.assertEqual(budget["headroom_to_monitor_mib"], 5553)

    def test_physical_free_memory_still_gates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "free=6143 MiB"):
            MODULE.assess_gpu_budget({"free_mib": 6143, "used_mib": 1000})

    def test_monitor_headroom_still_gates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "headroom_to_monitor=5119 MiB"):
            MODULE.assess_gpu_budget({"free_mib": 9000, "used_mib": 9217})

    def test_exact_boundaries_are_admitted(self) -> None:
        budget = MODULE.assess_gpu_budget({"free_mib": 6144, "used_mib": 9216})
        self.assertEqual(budget["headroom_to_monitor_mib"], 5120)


if __name__ == "__main__":
    unittest.main()

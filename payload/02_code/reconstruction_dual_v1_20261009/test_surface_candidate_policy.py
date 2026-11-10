"""Regress independent surface eligibility and support-preserving local gains."""

import ast
from pathlib import Path
import unittest

from surface_candidate_policy import select_surface_candidate


def candidate(support, triangles, coherent):
    return {"sparse_support_fraction_within_5cm": support, "triangle_count": triangles,
            "dominant_component_area_fraction": .995 if coherent else .80,
            "component_count": 80 if coherent else 20000}


def native_selector():
    source = Path(__file__).resolve().parents[2] / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/run_reliable_lidar_tsdf.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [node for node in tree.body if (isinstance(node, ast.Assign) and
            all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)) or
            (isinstance(node, ast.FunctionDef) and node.name == "select_surface_candidate")]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), scope)
    return scope["select_surface_candidate"]


class SurfaceCandidatePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = staticmethod(native_selector())

    def choose(self, f, v, b):
        return select_surface_candidate(f, v, b, self.original)[0]

    def test_broad_admission_does_not_depend_on_ineligible_strict_candidate(self):
        self.assertEqual(self.choose(candidate(.963, 1800, False), candidate(.944, 1000, True),
                                    candidate(.952, 1075, True)), "pgsr_depth_broad")

    def test_local_support_gain_is_not_discarded_by_total_triangle_floor(self):
        self.assertEqual(self.choose(candidate(.972, 1800, False), candidate(.960, 1000, True),
                                    candidate(.964, 1046, True)), "pgsr_depth_broad")

    def test_subthreshold_broad_support_is_not_accepted(self):
        self.assertEqual(self.choose(candidate(.963, 1800, False), candidate(.940, 1000, True),
                                    candidate(.949, 1075, True)), "registered_lidar_fused")

    def test_incoherent_broad_candidate_is_not_accepted(self):
        self.assertEqual(self.choose(candidate(.963, 1800, False), candidate(.951, 1000, True),
                                    candidate(.970, 1075, False)), "pgsr_depth_only")

    def test_excessive_surface_expansion_is_not_accepted(self):
        self.assertEqual(self.choose(candidate(.963, 1800, False), candidate(.951, 1000, True),
                                    candidate(.970, 1500, True)), "pgsr_depth_only")

    def test_coherent_sensor_fusion_retains_priority(self):
        self.assertEqual(self.choose(candidate(.963, 1800, True), candidate(.960, 1000, True),
                                    candidate(.970, 1075, True)), "registered_lidar_fused")

    def test_a_small_triangle_gain_without_support_gain_retains_strict_surface(self):
        self.assertEqual(self.choose(candidate(.963, 1800, False), candidate(.960, 1000, True),
                                    candidate(.960, 1046, True)), "pgsr_depth_only")

    def test_nonfinite_support_is_rejected(self):
        with self.assertRaises(ValueError):
            self.choose(candidate(.963, 1800, False), candidate(.960, 1000, True), candidate(float("nan"), 1046, True))


if __name__ == "__main__":
    unittest.main()

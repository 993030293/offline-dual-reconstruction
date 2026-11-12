"""Check that the offline free-flight viewer is packaged without scene payloads."""

import json
from pathlib import Path
import sys
import unittest

PAYLOAD = Path(__file__).resolve().parents[2]
VIEWER_CODE = PAYLOAD / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(VIEWER_CODE))

from build_raw_viewer import (
    SOURCE,
    SOURCE_ROOT,
    bind_packed_asset_budget,
    bind_viewer_page,
)


class PackagedViewerTemplateTests(unittest.TestCase):
    def test_template_contains_code_vendor_and_licenses_without_scene_geometry(self):
        required_page = {
            "index.html",
            "viewer.js",
            "calibrated_camera.js",
            "lazy_models.js",
            "motion_resolution.js",
            "packed_splat_asset.js",
            "persistent_asset_cache.js",
            "render_filter_contract.js",
            "render_schedule.js",
        }
        required_vendor = {
            "OrbitControls.js",
            "SPARK-LICENSE.txt",
            "spark.module.js",
            "THREE-LICENSE.txt",
            "three.core.js",
            "three.module.js",
            "postprocessing/Pass.js",
        }
        self.assertEqual(
            {path.name for path in SOURCE.iterdir() if path.is_file()}, required_page
        )
        vendor = SOURCE_ROOT / "shared-build/vendor"
        self.assertEqual(
            {path.relative_to(vendor).as_posix() for path in vendor.rglob("*") if path.is_file()},
            required_vendor,
        )
        self.assertFalse(any(SOURCE_ROOT.rglob("*.ply")))
        provenance = json.loads(
            (SOURCE_ROOT.parent / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["geometry_payloads_copied"], False)
        self.assertEqual(provenance["template_file_count"], 16)

    def test_current_template_budget_and_page_rebind_are_idempotent(self):
        packed = (SOURCE / "packed_splat_asset.js").read_text(encoding="utf-8")
        self.assertEqual(bind_packed_asset_budget(packed), packed)
        page = bind_viewer_page(
            (SOURCE / "index.html").read_text(encoding="utf-8"),
            "fixture case · 优化模型",
            1198,
            "c0_july-optimized",
        )
        self.assertIn("<title>fixture case · 优化模型 · 自由飞行预览</title>", page)
        self.assertIn("<h1>fixture case · 优化模型 · 1198 帧</h1>", page)
        self.assertEqual(page.count('value="c0_july-optimized" selected'), 1)


if __name__ == "__main__":
    unittest.main()

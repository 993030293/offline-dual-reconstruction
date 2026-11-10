"""Stage unchanged browser helpers against completed fresh full-model assets."""

import argparse
from pathlib import Path
import shutil
import sys

from atomic_json_windows import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--name", default="browser_module_audit_001")
    args = parser.parse_args()
    target = args.campaign / "viewer" / args.name
    target.mkdir(exist_ok=False)
    paths = [V3 / name for name in ("behind_camera_draw.js", "ordering_scratch_pool.js",
        "ordering_scratch_native_v2.js", "orbit_convergence_dag_v2.js")]
    paths += [Path(__file__).with_name("browser_module_audit.js")]
    for path in paths:
        shutil.copyfile(path, target / path.name)
        if path.read_bytes() != (target / path.name).read_bytes():
            raise AssertionError("The complete executed browser helper source differs")
    shutil.copyfile(args.campaign / "viewer/shared-build/vendor/SPARK-LICENSE.txt", target / "SPARK-LICENSE.txt")
    branches = []
    for case in read_json(args.campaign / "CASES.json")["cases"]:
        for mode in ("baseline", "optimized"):
            branch = case["case"] + "-" + mode
            folder = args.campaign / "viewer" / branch
            if not (folder / "ASSET_MANIFEST.json").exists():
                continue
            manifest = read_json(folder / "ASSET_MANIFEST.json")
            if not {"scene", "box"}.issubset(manifest):
                continue
            branches.append({"branch": branch, "full_asset": manifest["scene"],
                "source_model": read_json(folder / "cases/packed/display_manifest.json")["source_model"],
                "camera_path": data_identity(folder / "cases/packed/camera_path.json"),
                "full_scene_only": True})
    atomic_json(target / "CONTRACT.json", {"sources": [data_identity(path) for path in paths], "branches": branches,
        "all_registered_camera_rows_for_orbit_state_check": True, "native_full_Spark_sort_worker_oracles": True,
        "complete_RGBA_oracle_for_provably_behind_full_model": True, "primary_viewers_untouched": True,
        "work_counts_and_compatibility_are_not_FPS_or_quality_acceptance": True})
    (target / "index.html").write_text('''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>本轮浏览模块验证</title><link rel="icon" href="data:"><script type="importmap">{"imports":{"three":"../shared-build/vendor/three.module.js","three/addons/":"../shared-build/vendor/","@sparkjsdev/spark":"../shared-build/vendor/spark.module.js"}}</script></head>
<body><button id="run">运行本轮浏览模块完整验证</button><canvas id="view" aria-label="模块验证场景"></canvas><pre id="result" role="status">等待执行</pre><script type="module" src="./browser_module_audit.js"></script></body></html>''', encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

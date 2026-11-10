"""Check actual native model identity and profiler transfer receipts."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from native_feature_zero import native_feature_zero

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001"


def state(model, torch):
    result = {}
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation",
                 "_exposure", "max_radii2D", "xyz_gradient_accum", "denom"):
        value = getattr(model, name)
        result[name] = {"dtype": str(value.dtype), "device": str(value.device),
                       "shape": list(value.shape), "stride": list(value.stride()),
                       "offset": value.storage_offset(), "grad": value.requires_grad,
                       "bytes": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
    result["auxiliary"] = {name: getattr(model, name) for name in ("active_sh_degree", "max_sh_degree",
        "optimizer_type", "percent_dense", "spatial_lr_scale", "exposure_mapping", "pretrained_exposures")}
    result["rng"] = {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                     "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                     "cpu": torch.get_rng_state().tolist(),
                     "cuda": [value.tolist() for value in torch.cuda.get_rng_state_all()]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud
        torch.cuda.set_device(0)
        torch.manual_seed(157001)
        np.random.seed(157001)
        random.seed(157001)
        sources = [Path(__file__), CODE / "native_feature_zero.py",
                   Path(metadata["reference_source"]) / "scene/gaussian_model.py"]
        identities = [data_identity(path) for path in sources]
        copies = args.output / "source_snapshot"
        copies.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
        rows = []
        for number, (count, degree, dtype) in enumerate(((31, 0, np.float32), (129, 1, np.float32),
                                                        (257, 3, np.float32), (65, 3, np.float64))):
            generator = np.random.default_rng(157100 + number)
            points = generator.normal(size=(count, 3)).astype(dtype)
            colors = generator.uniform(size=(count, 3)).astype(dtype)
            if number == 3:
                points = np.column_stack((points, points))[:, ::2]
                colors = np.column_stack((colors, colors))[:, ::2]
            cloud = BasicPointCloud(points, colors, np.zeros((count, 3), dtype=np.float32))
            values, transfers = {}, {}
            for mode in ("native", "direct"):
                leaf = args.output / (f"t{number}_" + mode)
                leaf.mkdir()
                with native_feature_zero(leaf, mode == "direct") as trace:
                    model = GaussianModel(degree)
                    model.create_from_pcd(cloud, [SimpleNamespace(image_name="irregular_view")], 1.25)
                    values[mode] = state(model, torch)
                    del model
                gc.collect()
                torch.cuda.empty_cache()
                transfers[mode] = trace
            checks = {"native_all_state_and_RNG_exact": values["native"] == values["direct"],
                      "both_native_hooks_restored": all(row["native_functions_restored"] for row in transfers.values()),
                      "all_actual_profiler_bytes_match_native_uploads": all(row["calls"][0]["profiler_matches_all_native_uploads"]
                                                                           and row["actual_profiler_HtoD_bytes"] > 0 for row in transfers.values()),
                      "direct_only_one_zero_upload_removed": len(transfers["native"]["calls"][0]["uploads"]) == 4
                                                                  and len(transfers["direct"]["calls"][0]["uploads"]) == 3,
                      "same_original_three_nonzero_uploads": transfers["native"]["calls"][0]["uploads"][:2]
                          + transfers["native"]["calls"][0]["uploads"][3:] == transfers["direct"]["calls"][0]["uploads"]}
            rows.append({"count": count, "degree": degree, "checks": checks, "states": values,
                         "transfers": transfers})
            atomic_json(args.output / "PARTIAL.json", {"rows": rows, "new_optimization_count": 0})
            if not all(checks.values()):
                raise AssertionError("The actual native zero-feature component contract failed")
        checks = {"all_four_irregular_native_cases": len(rows) == 4 and all(all(row["checks"].values()) for row in rows),
                  "physical_sources_unchanged": identities == [data_identity(path) for path in sources]}
        atomic_json(args.output / "RESULT.json", {"status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "rows": rows, "sources": identities, "runtime": metadata,
            "extra_actual_HtoD_bytes": sum(trace["actual_profiler_HtoD_bytes"] for row in rows for trace in row["transfers"].values()),
            "new_optimization_count": 0, "full_quality_GT_training": "not_run"})
        if not all(checks.values()):
            raise AssertionError("A native zero-feature source check failed")


if __name__ == "__main__":
    main()

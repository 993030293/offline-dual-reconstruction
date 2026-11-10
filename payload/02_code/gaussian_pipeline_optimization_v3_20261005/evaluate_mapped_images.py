"""Use the original renderer and every fixed full/target development metric."""

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
import evaluate_pair
from mapped_image_linux_guard import LinuxRSSGuard


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.pair, args.output):
        path.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json) as guard:
        local = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_evaluation_localization_002/H_LOCALIZATION.json"
        paths = [Path(__file__), V1/"evaluate_pair.py", V1/"full_intrinsics.py", V1/"runtime.py",
            Path(__file__).with_name("mapped_image_linux_guard.py")]
        sources = [data_identity(path) for path in paths]
        localization = read(local)
        if len(localization["rows"]) != 24 or any(row["status"] != "localized" for row in localization["rows"]):
            raise ValueError("All original24 development cameras must remain localized")
        models = {}
        for mode in ("native", "mapped"):
            output = args.pair/mode/"training"
            state = read(output/"STATE.json")
            result = read(output/"MAPPED_TRAINING_RESULT.json")
            if state["status"] != "completed" or state["iteration"] != 12000 or state["seed"] != args.seed or not all(result["checks"].values()):
                raise ValueError("Both current native12000 models must finish before evaluation")
            models[mode] = data_identity(output/"model/point_cloud/iteration_12000/point_cloud.ply")
        source = inspect.getsource(evaluate_pair.main)
        anchor = "args.pair/label/"
        if source.count(anchor) != 3:
            raise ValueError("The original model, input and state path boundary changed")
        source = source.replace(anchor, 'args.pair/label/"training"/')
        namespace = dict(evaluate_pair.main.__globals__)
        exec(compile(source, "<r131_original_full24_metrics_native_training_subdirectory>", "exec"), namespace)
        previous = sys.argv
        try:
            sys.argv = [str(V1/"evaluate_pair.py"), "--runtime", str(ROOT/"04_outputs/box2_3dgs_display_v1_20260930"),
                "--reference-run", str(ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330"),
                "--pair", str(args.pair), "--control-label", "native", "--candidate-label", "mapped", "--iterations", "12000",
                "--evaluation-folder", "evaluation_seed"+str(args.seed), "--localization", str(local), "--input-version", "new_map_admission_v1"]
            namespace["main"]()
        finally:
            sys.argv = previous
        native = read(args.pair/("evaluation_seed"+str(args.seed))/"EVALUATION.json")
        rows = {mode: {row["image_name"]: row for row in native["results"][mode]["rows"]} for mode in models}
        deltas = []
        for name in sorted(row["image_name"] for row in localization["rows"]):
            for region in ("full_image", "target"):
                before, after = rows["native"][name].get(region), rows["mapped"][name].get(region)
                if before is None or after is None:
                    raise ValueError("A full24 original full/target metric disappeared")
                row = {"image_name": name, "region": region, "PSNR_delta_dB": after["PSNR_dB"]-before["PSNR_dB"],
                    "SSIM_delta": after["SSIM"]-before["SSIM"]}
                row["passed"] = row["PSNR_delta_dB"] >= -.1 and row["SSIM_delta"] >= -.002
                deltas.append(row)
        checks = {"all_original24_full_target_rows": len(deltas) == 48,
            "all_original_per_view_quality_protections": all(row["passed"] for row in deltas),
            "original_mean_quality_protection": native["development_metric_guard_passed"],
            "same_original_sources": sources == [data_identity(path) for path in paths],
            "models_preserved": models == {mode: data_identity(args.pair/mode/"training/model/point_cloud/iteration_12000/point_cloud.ply") for mode in models},
            "argv_restored": sys.argv is previous}
        guard.sample()
        result = {"status": "complete_seed_full24_quality_passed" if all(checks.values()) else "failed_original_quality_protection",
            "seed": args.seed, "checks": checks, "deltas": deltas, "models": models,
            "sources": sources, "localization": data_identity(local), "native_evaluation": native,
            "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0,
            "strict_blind_or_unseen_object_or_GT_claim": False}
        atomic_json(args.output/"RESULT.json", result)
        print(json.dumps({"status": result["status"], "seed": args.seed,
            "failed_rows": sum(not row["passed"] for row in deltas), "checks": checks}))
        if not all(checks.values()):
            raise AssertionError("An original complete full24 per-view quality gate failed")


if __name__ == "__main__":
    main()

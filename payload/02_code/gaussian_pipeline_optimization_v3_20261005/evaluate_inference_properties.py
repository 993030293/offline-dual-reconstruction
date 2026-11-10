"""Run the complete protected evaluator with a private native property subclass."""

import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import evaluate_interior_ssim
import evaluate_view_major


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "cache"), required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--fault-render", type=int, default=0)
    args = parser.parse_args()
    output = args.pair.parent
    render_output = output / "render_arrays"
    render_output.mkdir(exist_ok=False)
    prior_activate, prior_argv = evaluate_interior_ssim.activate_reference, sys.argv
    installed, workspace, render_rows, sources = {}, None, [], []

    def activate(runtime):
        nonlocal workspace
        metadata = prior_activate(runtime)
        import numpy as np
        import torch
        import gaussian_renderer
        import scene.gaussian_model as gaussian_module
        from inference_property_workspace import InferencePropertyWorkspace
        installed.update(module=gaussian_module, native_class=gaussian_module.GaussianModel,
                         renderer=gaussian_renderer, render=gaussian_renderer.render)
        workspace = InferencePropertyWorkspace(installed["native_class"], args.mode == "cache")
        gaussian_module.GaussianModel = workspace.model_class

        def render(*values, **options):
            ordinal = len(render_rows)
            if args.fault_render and ordinal + 1 == args.fault_render:
                raise RuntimeError("REGISTERED_R139_RENDER_FAILURE")
            result = installed["render"](*values, **options)
            actual = result["render"].detach().contiguous().cpu().numpy()
            path = render_output / f"{ordinal:03d}.npy"
            np.save(path, actual, allow_pickle=False)
            render_rows.append({"ordinal": ordinal, "array": data_identity(path),
                                "shape": list(actual.shape), "dtype": str(actual.dtype)})
            return result

        gaussian_renderer.render = render
        sources.extend(data_identity(path) for path in [Path(__file__), CODE / "inference_property_workspace.py",
            Path(gaussian_renderer.__file__), Path(gaussian_module.__file__), CODE / "evaluate_view_major.py",
            CODE / "evaluate_interior_ssim.py", CODE / "view_major_evaluation.py"])
        return metadata

    failure = None
    evaluate_interior_ssim.activate_reference = activate
    try:
        sys.argv = [str(CODE / "evaluate_view_major.py"), "--mode", "view", "--pair", str(args.pair),
                    "--localization", str(args.localization)]
        evaluate_view_major.main()
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        evaluate_interior_ssim.activate_reference, sys.argv = prior_activate, prior_argv
        if installed:
            installed["module"].GaussianModel = installed["native_class"]
            installed["renderer"].render = installed["render"]
        if workspace is not None:
            workspace.close()
        restored = bool(installed and installed["module"].GaussianModel is installed["native_class"]
                        and installed["renderer"].render is installed["render"]
                        and evaluate_interior_ssim.activate_reference is prior_activate and sys.argv is prior_argv)
        if failure is not None:
            atomic_json(output / "FAILURE.json", {"status": "actual_error_propagated", "error": failure,
                "render_rows": render_rows, "workspace": workspace.trace if workspace else None,
                "bindings_restored": restored, "sources": sources,
                "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
    import torch
    measurement = json.loads((output / "measurement/RESULT.json").read_text())
    evaluation = json.loads((args.pair / "evaluation/EVALUATION.json").read_text())
    trace = workspace.trace
    checks = {"complete48_original_render_arrays": len(render_rows) == 48,
        "all192_original_property_requests_arrived": all(value == 48 for value in trace["requests"].values()),
        "all_original_full_input_model_RNG_cache_protections": all(measurement["checks"].values()),
        "complete24_original_quality_guard": evaluation["development_metric_guard_passed"]
            and evaluation["all_frames_included_in_denominator"] == 24,
        "all_source_identities_unchanged": sources == [data_identity(row["path"]) for row in sources],
        "all_private_bindings_restored": restored,
        "owned_workspace_closed_bounded_and_drained": trace["closed"] and trace["resident_bytes_after_close"] == 0
            and trace["peak_resident_bytes"] <= trace["maximum_bytes"],
        "actual_CUDA_elements_measured_for_each_property": all(value > 0 for value in trace["native_CUDA_elements"].values()),
        "CUDA_reserved_below12GiB": torch.cuda.max_memory_reserved() <= 12 * 1024**3}
    if args.mode == "cache":
        checks["exactly_two_native_executions_per_property_no_fallback"] = (all(value == 2 for value in trace["native_executions"].values())
            and trace["invalidations"] == 0 and set(trace["routes"]) == {"first_native_result"})
    result = {"status": "complete_evaluation_protected" if all(checks.values()) else "failed_verification",
        "mode": args.mode, "checks": checks, "workspace": trace, "render_rows": render_rows, "sources": sources,
        "measurement": str(output / "measurement/RESULT.json"), "complete_child_seconds": time.perf_counter() - STARTED,
        "actual_CUDA_reserved_peak_bytes": torch.cuda.max_memory_reserved(), "new_optimization_count": 0}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "workspace": trace}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The complete fixed-model property consumer failed a protection")


if __name__ == "__main__":
    main()

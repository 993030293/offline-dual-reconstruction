"""Inject registered real failures and verify original evaluator restoration."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "interior"), required=True)
    parser.add_argument("--fault", choices=("error", "cancel"), required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    args = parser.parse_args()
    metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
    import numpy as np
    import random
    import torch
    import torch.nn.functional as functional
    from utils import loss_utils
    from scene.gaussian_model import GaussianModel
    import evaluate_pair
    import evaluate_device_artifacts
    import evaluate_interior_conv_work as entry

    def bindings():
        return (evaluate_pair.main, evaluate_pair.activate_reference,
                evaluate_device_artifacts.evaluate_dag_models, loss_utils.ssim,
                GaussianModel.load_ply, functional.conv2d)

    def rng():
        return (repr(random.getstate()), repr(np.random.get_state()), torch.random.get_rng_state().tolist(),
                [v.tolist() for v in torch.cuda.get_rng_state_all()])

    random.seed(97531)
    np.random.seed(97531)
    torch.manual_seed(97531)
    before_rng, original = rng(), bindings()
    previous = sys.argv
    original_activate = entry.entry.activate_reference
    sys.argv = [str(Path(entry.__file__)), "--mode", args.mode, "--pair", str(args.pair),
                "--localization", str(args.localization)]
    error = None
    try:
        entry.entry.activate_reference = lambda runtime: metadata
        entry.run(fault=args.fault)
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        entry.entry.activate_reference = original_activate
        sys.argv = previous
    import json
    work = json.loads((args.pair.parent / "CONV_WORK.json").read_text(encoding="utf-8"))
    frozen = (args.pair.parent / "ACTUAL_MEASUREMENT_SOURCE.py").read_bytes()
    same_directory = False
    sys.argv = [str(Path(entry.__file__)), "--mode", args.mode, "--pair", str(args.pair),
                "--localization", str(args.localization)]
    try:
        entry.run(fault=args.fault)
    except FileExistsError:
        same_directory = True
    finally:
        sys.argv = previous
    expected = {"type": "RuntimeError", "message": "owned_R134_V3_fifth_SSIM_failure"}
    checks = {"expected_real_error_or_SIGINT_propagated": error == expected if args.fault == "error"
                  else error is not None and error["type"] == "KeyboardInterrupt",
              "original_all_six_bindings_restored": bindings() == original,
              "preselected_metadata_alias_restored": entry.entry.activate_reference is original_activate,
              "original_global_CPU_and_CUDA_RNG_restored": rng() == before_rng,
              "actual_census_and_caches_closed": all(work["checks"].values()),
              "actual_numerical_work_prefix_present": work["trace"]["successful_CUDA_output_elements"] > 0,
              "same_directory_rejected_without_overwrite": same_directory
                  and frozen == (args.pair.parent / "ACTUAL_MEASUREMENT_SOURCE.py").read_bytes(),
              "original_argv_restored": sys.argv is previous}
    atomic_json(args.pair.parent / "FAULT_RESULT.json", {"status": "expected_failure_cleanup_verified" if all(checks.values()) else "failed",
        "checks": checks, "error": error, "fault": args.fault, "mode": args.mode,
        "script": data_identity(Path(__file__)), "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("An original complete evaluator failure restoration changed")
    raise SystemExit(86)


if __name__ == "__main__":
    main()


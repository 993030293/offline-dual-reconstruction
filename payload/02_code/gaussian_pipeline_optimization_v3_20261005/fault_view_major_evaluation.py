"""Inject an owned error or real SIGINT and verify evaluator cleanup in process."""

import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import sys

import evaluate_view_major


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "view"), required=True)
    parser.add_argument("--fault", choices=("error", "cancel"), required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    args = parser.parse_args()
    registry, initial = [], {}

    def recorded_factory(cache_class, factory):
        import evaluate_pair
        import evaluate_device_artifacts
        from utils import loss_utils
        from scene.gaussian_model import GaussianModel
        import numpy as np
        import random
        import torch
        initial.update(pair_main=evaluate_pair.main, pair_activate=evaluate_pair.activate_reference,
                       dag=evaluate_device_artifacts.evaluate_dag_models, ssim=loss_utils.ssim, reader=GaussianModel.load_ply,
                       rng={"python": repr(random.getstate()), "numpy": repr(np.random.get_state()),
                            "CPU": torch.random.get_rng_state().tolist(), "CUDA": [state.tolist() for state in torch.cuda.get_rng_state_all()]})

        class OwnedFaultReference(cache_class):
            def __init__(self):
                super().__init__(factory)
                self.attempted = self.completed = self.closed = 0

            def __call__(self, *arguments, **keywords):
                self.attempted += 1
                if self.attempted == 5:
                    if args.fault == "cancel":
                        os.kill(os.getpid(), signal.SIGINT)
                        raise AssertionError("The real owned SIGINT did not interrupt the evaluator")
                    raise RuntimeError("owned_R135_fifth_SSIM_failure")
                result = super().__call__(*arguments, **keywords)
                self.completed += 1
                return result

            def clear(self):
                super().clear()
                self.closed += 1

        cache = OwnedFaultReference()
        registry.append(cache)
        return cache

    def fault_bound(namespace, source):
        anchor = "inner = ReferenceSSIM(loss_utils.create_window)"
        if source.count(anchor) != 1:
            raise ValueError("The fixed original reference factory changed before failure injection")
        source = source.replace(anchor, "inner = _recorded_factory(ReferenceSSIM, loss_utils.create_window)")
        destination = args.pair.parent / "FAULT_COMPOSED_MEASUREMENT.py"
        destination.write_text(source, encoding="utf-8")
        namespace["_recorded_factory"] = recorded_factory
        exec(compile(source, "<owned-failure-current-view-major-measurement>", "exec"), namespace)
        namespace["main"]()

    source = inspect.getsource(evaluate_view_major.main)
    anchor = '    namespace["main"]()'
    if source.count(anchor) != 1:
        raise ValueError("The actual evaluator runner source changed")
    source = source.replace(anchor, "    _fault_bound(namespace, source)")
    namespace = dict(vars(evaluate_view_major))
    namespace["_fault_bound"] = fault_bound
    exec(compile(source, "<owned-failure-view-major-runner>", "exec"), namespace)
    owned_main = namespace["main"]
    old_argv = sys.argv
    sys.argv = [str(Path(evaluate_view_major.__file__)), "--mode", args.mode, "--pair", str(args.pair), "--localization", str(args.localization)]
    error = None
    try:
        owned_main()
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
    if error is None:
        raise AssertionError("The owned mid-evaluation failure was not executed")
    import evaluate_pair
    import evaluate_device_artifacts
    from utils import loss_utils
    from scene.gaussian_model import GaussianModel
    import numpy as np
    import random
    import torch
    torch.cuda.synchronize()
    after_rng = {"python": repr(random.getstate()), "numpy": repr(np.random.get_state()),
                 "CPU": torch.random.get_rng_state().tolist(), "CUDA": [state.tolist() for state in torch.cuda.get_rng_state_all()]}
    checks = {"actual_expected_error_or_signal_propagated": (error["type"] == "KeyboardInterrupt" if args.fault == "cancel"
              else error == {"type": "RuntimeError", "message": "owned_R135_fifth_SSIM_failure"}),
              "exactly_four_original_calls_before_failure": len(registry) == 1 and registry[0].attempted == 5 and registry[0].completed == 4,
              "all_original_cache_entries_and_windows_released": len(registry) == 1 and registry[0].closed >= 1
                  and not registry[0].entries and not registry[0].windows and registry[0].resident_bytes == 0,
              "all_original_function_bindings_restored": evaluate_pair.main is initial["pair_main"]
                  and evaluate_pair.activate_reference is initial["pair_activate"] and loss_utils.ssim is initial["ssim"]
                  and GaussianModel.load_ply is initial["reader"] and evaluate_device_artifacts.evaluate_dag_models is initial["dag"],
              "all_temporary_helpers_removed": not hasattr(evaluate_pair, "_observe_gaussian") and not hasattr(evaluate_pair, "_finish_view"),
              "all_original_RNG_unchanged": initial["rng"] == after_rng,
              "real_early_contract_preserved_no_final_completed_result": (args.pair / "evaluation/EVALUATION_CONTRACT.json").is_file()
                  and not (args.pair / "evaluation/EVALUATION.json").exists()}
    try:
        owned_main()
    except FileExistsError:
        checks["same_owned_output_retry_rejected_originally"] = True
    except BaseException:
        checks["same_owned_output_retry_rejected_originally"] = False
    else:
        checks["same_owned_output_retry_rejected_originally"] = False
    finally:
        sys.argv = old_argv
    result = {"status": "expected_failure_cleanup_verified" if all(checks.values()) else "failed_lifecycle",
              "checks": checks, "mode": args.mode, "fault": args.fault, "actual_exception": error,
              "closed_original_cache_receipt": registry[0].receipt(), "new_optimization_count": 0,
              "intentional_child_exit_code": 86, "training_checkpoint_recovery_claim": False}
    from runtime import atomic_json
    atomic_json(args.pair.parent / "FAULT_RESULT.json", result)
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise AssertionError("An original evaluator binding, cache, RNG or retry contract failed")
    raise SystemExit(86)


if __name__ == "__main__":
    main()

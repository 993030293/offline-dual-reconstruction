"""Evaluate the unchanged Q30/Q61/R100 entry with private SSIM measurement."""

import time

STARTED = time.perf_counter()

import argparse
import hashlib
import inspect
import json
import linecache
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "interior"), required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    args = parser.parse_args()
    diagnostic = args.pair.parent / "measurement"
    diagnostic.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(diagnostic / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": owner.pid, "creation_time": owner.create_time()})
    with LinuxRSSGuard(diagnostic, atomic_json, maximum_seconds=300, maximum_bytes=8 * 1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        import torch
        from utils import loss_utils
        from scene.gaussian_model import GaussianModel
        from reference_ssim import ReferenceSSIM
        from interior_ssim_reuse import InteriorSSIM
        import evaluate_pair
        import evaluate_dag_models
        import evaluate_device_artifacts
        random.seed(97531)
        np.random.seed(97531)
        torch.manual_seed(97531)

        def rng():
            return {"Python": repr(random.getstate()), "NumPy": repr(np.random.get_state()),
                    "Torch_CPU": torch.random.get_rng_state().tolist(),
                    "all_CUDA": [state.tolist() for state in torch.cuda.get_rng_state_all()]}

        before_rng = rng()
        rows, states = [], []
        composition_started = time.perf_counter()
        inner = ReferenceSSIM(loss_utils.create_window) if args.mode == "native" else InteriorSSIM(loss_utils.create_window)
        lifecycle = {"state_validation_seconds": 0.0}

        def measured(prediction, reference, window_size=11, size_average=True):
            torch.cuda.synchronize()
            started = time.perf_counter()
            left, right = prediction.clone(), reference.clone()
            versions = (prediction._version, reference._version)
            result = inner(prediction, reference, window_size, size_average)
            unchanged = (torch.equal(prediction.contiguous().view(torch.uint8), left.contiguous().view(torch.uint8))
                         and torch.equal(reference.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
                         and (prediction._version, reference._version) == versions)
            torch.cuda.synchronize()
            rows.append({"shape": list(prediction.shape), "prediction_stride": list(prediction.stride()),
                         "reference_stride": list(reference.stride()), "input_bits_and_versions_unchanged": unchanged,
                         "seconds_including_complete_input_protection": time.perf_counter() - started})
            if not unchanged:
                raise AssertionError("An actual SSIM call mutated original complete input bits")
            return result

        class MeasuredCache:
            __call__ = staticmethod(measured)

            def receipt(self):
                return inner.receipt()

            def clear(self):
                inner.clear()

        captured = {}

        def observe(gaussian, stage):
            started = time.perf_counter()
            fields = {}
            for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
                tensor = getattr(gaussian, name)
                data = tensor.detach().contiguous().cpu().numpy().tobytes()
                fields[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "stride": list(tensor.stride()),
                                "version": tensor._version, "requires_grad": tensor.requires_grad,
                                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if stage == "before":
                captured[id(gaussian)] = fields
            else:
                expected = captured.pop(id(gaussian))
                same = expected == fields
                states.append({"fields": fields, "all_original_model_fields_unchanged": same})
                if not same:
                    raise AssertionError("The evaluation changed a complete original Gaussian parameter")
            lifecycle["state_validation_seconds"] += time.perf_counter() - started

        pair_source = inspect.getsource(evaluate_pair.main)
        load_anchor = "        gaussian.load_ply(str(model_path))"
        delete_anchor = "        del gaussian"
        if pair_source.count(load_anchor) != 1 or pair_source.count(delete_anchor) != 1:
            raise ValueError("The existing complete Gaussian evaluation lifecycle changed")
        pair_namespace = dict(vars(evaluate_pair))
        pair_namespace["_observe_gaussian"] = observe
        pair_source = pair_source.replace(load_anchor, load_anchor + '\n        _observe_gaussian(gaussian, "before")')
        pair_source = pair_source.replace(delete_anchor, '        _observe_gaussian(gaussian, "after")\n' + delete_anchor)
        exec(compile(pair_source, "<private-current-evaluator-model-protection>", "exec"), pair_namespace)
        dag_source = inspect.getsource(evaluate_dag_models.main)
        cache_anchor = "    cache = ReferenceSSIM(loss_utils.create_window)"
        if dag_source.count(cache_anchor) != 1:
            raise ValueError("The current Q30 evaluator factory changed")
        dag_namespace = dict(vars(evaluate_dag_models))
        dag_namespace["_measured_factory"] = lambda factory: MeasuredCache()
        dag_namespace["activate_reference"] = lambda runtime: metadata
        dag_source = dag_source.replace(cache_anchor, "    cache = _measured_factory(loss_utils.create_window)")
        exec(compile(dag_source, "<private-current-evaluator-measured-SSIM>", "exec"), dag_namespace)
        old_pair_main, old_dag, old_argv = evaluate_pair.main, evaluate_device_artifacts.evaluate_dag_models, sys.argv
        original_bindings = (evaluate_pair.activate_reference, loss_utils.ssim, GaussianModel.load_ply)
        paths = [Path(__file__), CODE / "interior_ssim_reuse.py", CODE / "evaluate_dag_models.py",
                 CODE / "evaluate_device_artifacts.py", CODE / "device_artifact_picture.py",
                 V1 / "evaluate_pair.py", V1 / "full_intrinsics.py", V2 / "reference_ssim.py",
                 V2 / "vectorized_gaussian_read.py", Path(loss_utils.__file__)]
        sources = [data_identity(path) for path in paths]
        copies = diagnostic / "source_copies"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        (diagnostic / "COMPOSED_PAIR.py").write_text(pair_source, encoding="utf-8")
        (diagnostic / "COMPOSED_DAG.py").write_text(dag_source, encoding="utf-8")
        private_filename = "<private-current-evaluator-model-protection>"
        if hasattr(evaluate_pair, "_observe_gaussian") or private_filename in linecache.cache:
            raise RuntimeError("The private evaluation helper namespace is already owned")
        linecache.cache[private_filename] = (len(pair_source), None, pair_source.splitlines(True), private_filename)
        composition_seconds = time.perf_counter() - composition_started
        try:
            evaluate_pair._observe_gaussian = observe
            evaluate_pair.main = pair_namespace["main"]
            evaluate_device_artifacts.evaluate_dag_models = SimpleNamespace(main=dag_namespace["main"], __file__=evaluate_dag_models.__file__)
            sys.argv = [str(CODE / "evaluate_device_artifacts.py"), "--mode", "device", "--runtime",
                str(ROOT / "04_outputs/box2_3dgs_display_v1_20260930"), "--pair", str(args.pair),
                "--localization", str(args.localization), "--reference-run",
                str(ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005"),
                "--input-version", "new_map_admission_v1", "--iterations", "12000"]
            evaluate_device_artifacts.main()
        finally:
            evaluate_pair.main, evaluate_device_artifacts.evaluate_dag_models, sys.argv = old_pair_main, old_dag, old_argv
            del evaluate_pair._observe_gaussian
            linecache.cache.pop(private_filename)
        torch.cuda.synchronize()
        cache_receipt = inner.receipt()
        checks = {"all96_actual_full_and_target_SSIM_inputs_exact": len(rows) == 96 and all(row["input_bits_and_versions_unchanged"] for row in rows),
                  "both_complete_actual_Gaussian_models_unchanged": len(states) == 2 and not captured
                      and all(row["all_original_model_fields_unchanged"] for row in states),
                  "global_CPU_and_all_CUDA_RNG_unchanged": before_rng == rng(),
                  "all_source_identities_unchanged": sources == [data_identity(path) for path in paths],
                  "original_function_bindings_restored": (evaluate_pair.activate_reference, loss_utils.ssim, GaussianModel.load_ply) == original_bindings
                      and evaluate_pair.main is old_pair_main and evaluate_device_artifacts.evaluate_dag_models is old_dag
                      and not hasattr(evaluate_pair, "_observe_gaussian") and private_filename not in linecache.cache}
        if args.mode == "interior":
            checks["all48_actual_target_crops_reused"] = cache_receipt["interior_reused"] == 48
        result = {"status": "complete_evaluation_protected" if all(checks.values()) else "failed_protection", "checks": checks,
                  "mode": args.mode, "sources": sources, "rows": rows, "model_states": states, "cache_before_clear": cache_receipt,
                  "composition_source_copy_identity_seconds": composition_seconds,
                  "complete_SSIM_seconds_including_input_protection": sum(row["seconds_including_complete_input_protection"] for row in rows),
                  "complete_model_state_validation_seconds": lifecycle["state_validation_seconds"],
                  "complete_child_seconds": time.perf_counter() - STARTED,
                  "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "new_optimization_count": 0, "reference_run_ignored_by_explicit_localization_current_entry": True}
        inner.clear()
        result["cache_after_clear"] = inner.receipt()
        checks["cache_clear_released_all_entries_and_last_map"] = result["cache_after_clear"]["resident_accounted_bytes"] == 0
        atomic_json(diagnostic / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks,
              "SSIM_seconds": result["complete_SSIM_seconds_including_input_protection"],
              "cache": cache_receipt, "complete_child_seconds": result["complete_child_seconds"]}), flush=True)
        if not all(checks.values()):
            raise AssertionError("An actual full evaluator input, state or lifecycle protection failed")


if __name__ == "__main__":
    main()

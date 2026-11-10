"""Keep view-major evaluation common while measuring exact crop-interior reuse."""

import inspect
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
import evaluate_interior_ssim as entry
from view_major_evaluation import compose_view_major, finish_reference_view
from ssim_conv_work import native_conv_work
from runtime import atomic_json, data_identity


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError("The current complete evaluator source changed: " + anchor[:100])
    return source.replace(anchor, replacement)


def finish_coupled_view(inner, trace):
    from interior_ssim_reuse import InteriorSSIM

    if isinstance(inner, InteriorSSIM):
        inner.last = None
        inner.reserve_last_map(0)
        target_cache = inner.cache
    else:
        target_cache = inner
    finish_reference_view(target_cache, trace)
    if isinstance(inner, InteriorSSIM) and inner.last is not None:
        raise AssertionError("The completed view retains a prediction map")


def composed_main():
    source = inspect.getsource(entry.main)
    source = replace_once(source, 'pair_source = inspect.getsource(evaluate_pair.main)',
                          'pair_source = compose_view_major(evaluate_pair.main)')
    source = replace_once(source, 'pair_namespace["_observe_gaussian"] = observe',
                          'pair_namespace["_observe_gaussian"] = observe\n        pair_namespace["_finish_view"] = finish_view')
    source = replace_once(source, 'evaluate_pair._observe_gaussian = observe',
                          'evaluate_pair._observe_gaussian = observe\n            evaluate_pair._finish_view = finish_view')
    source = replace_once(source, 'del evaluate_pair._observe_gaussian',
                          'del evaluate_pair._observe_gaussian\n            del evaluate_pair._finish_view')
    source = replace_once(source, 'composition_started = time.perf_counter()',
                          'composition_started = time.perf_counter()\n        finished_views = []')
    source = replace_once(source, 'lifecycle = {"state_validation_seconds": 0.0}',
        'lifecycle = {"state_validation_seconds": 0.0}\n'
        '        _register_inner(inner)\n'
        '        def finish_view():\n'
        '            finish_coupled_view(inner, finished_views)')
    source = replace_once(source, 'rows.append({"shape": list(prediction.shape)',
                          'rows.append({"native_conv_calls_before_return": len(_conv_trace["rows"]), "shape": list(prediction.shape)')
    source = replace_once(source, 'result = inner(prediction, reference, window_size, size_average)',
                          '_fault_before_SSIM(len(rows) + 1)\n            result = inner(prediction, reference, window_size, size_average)')
    source = replace_once(source, '"original_function_bindings_restored":',
                          '"all24_reference_views_released": len(finished_views) == 24 and not hasattr(evaluate_pair, "_finish_view"),\n                  "original_function_bindings_restored":')
    source = replace_once(source, '"mode": args.mode, "sources": sources, "rows": rows, "model_states": states, "cache_before_clear": cache_receipt,',
                          '"mode": args.mode, "sources": sources, "rows": rows, "model_states": states, "cache_before_clear": cache_receipt, "finished_reference_views": finished_views,')
    source = replace_once(source, 'CODE / "interior_ssim_reuse.py", CODE / "evaluate_dag_models.py",',
                          'CODE / "interior_ssim_reuse.py", CODE / "evaluate_interior_ssim.py", CODE / "view_major_evaluation.py", CODE / "ssim_conv_work.py", CODE / "evaluate_dag_models.py",')
    return source


def run(fault=None):
    pair = Path(sys.argv[sys.argv.index("--pair") + 1])
    pair.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/iscw001")
    source = composed_main()
    source_path = pair.parent / "ACTUAL_MEASUREMENT_SOURCE.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("x", encoding="utf-8") as stream:
        stream.write(source)
    registry = []

    def fault_before(ordinal):
        if ordinal == 5 and fault == "error":
            raise RuntimeError("owned_R134_V3_fifth_SSIM_failure")
        if ordinal == 5 and fault == "cancel":
            import os
            import signal
            os.kill(os.getpid(), signal.SIGINT)
            raise AssertionError("The owned SIGINT did not interrupt evaluation")

    namespace = dict(vars(entry))
    namespace.update(__file__=str(Path(__file__)), STARTED=STARTED,
                     compose_view_major=compose_view_major, finish_reference_view=finish_reference_view,
                     finish_coupled_view=finish_coupled_view,
                     _register_inner=registry.append, _fault_before_SSIM=fault_before)
    error = None
    with native_conv_work() as trace:
        namespace["_conv_trace"] = trace
        try:
            exec(compile(source, "<private-view-major-interior-conv-work>", "exec"), namespace)
            namespace["main"]()
        except BaseException as failure:
            error = failure
        finally:
            for inner in registry:
                inner.clear()
    from utils import loss_utils
    from scene.gaussian_model import GaussianModel
    import evaluate_pair
    import torch
    torch.cuda.synchronize()
    checks = {"native_conv_binding_restored": trace["original_binding_restored"],
              "private_helper_bindings_removed": not hasattr(evaluate_pair, "_observe_gaussian")
                  and not hasattr(evaluate_pair, "_finish_view"),
              "all_owned_caches_and_last_maps_closed": bool(registry) and all(
                  inner.receipt()["resident_accounted_bytes"] == 0
                  and inner.receipt().get("retained_last_map_bytes", 0) == 0 for inner in registry),
              "actual_convolution_work_nonzero": trace["successful_CUDA_output_elements"] > 0}
    atomic_json(pair.parent / "CONV_WORK.json", {"trace": trace, "checks": checks,
        "fault": fault, "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        "composed_measurement_source": data_identity(source_path),
        "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
    if error is not None:
        raise error
    if not all(checks.values()):
        raise AssertionError("A convolution work ownership or binding protection failed")


if __name__ == "__main__":
    run()


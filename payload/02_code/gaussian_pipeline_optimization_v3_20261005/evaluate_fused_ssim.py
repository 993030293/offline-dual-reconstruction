"""Measure a fused formula against the same complete interior-reuse evaluator."""

import os
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
import evaluate_interior_conv_work as original
from runtime import activate_reference, atomic_json, data_identity

ROOT, CODE = original.ROOT, original.CODE
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup020/R172"
entry = original.entry


def composed_main():
    source = original.composed_main()
    source = original.replace_once(source,
        'from interior_ssim_reuse import InteriorSSIM',
        'from interior_ssim_reuse import InteriorSSIM\n        from fused_ssim_formula import FusedInteriorSSIM\n        from ssim_output_census import output_census')
    source = original.replace_once(source,
        'inner = ReferenceSSIM(loss_utils.create_window) if args.mode == "native" else InteriorSSIM(loss_utils.create_window)',
        'inner = InteriorSSIM(loss_utils.create_window) if args.mode == "native" else FusedInteriorSSIM(loss_utils.create_window)')
    source = original.replace_once(source,
        '            result = inner(prediction, reference, window_size, size_average)',
        '            with output_census(_output_trace):\n                result = inner(prediction, reference, window_size, size_average)')
    source = original.replace_once(source, 'if args.mode == "interior":', 'if args.mode in ("native", "interior"):')
    source = original.replace_once(source,
        '"seconds_including_complete_input_protection": time.perf_counter() - started}',
        '"result_dtype": str(result.dtype), "result_shape": list(result.shape),\n'
        '                         "result_values": result.detach().cpu().tolist(),\n'
        '                         "seconds_including_complete_input_protection": time.perf_counter() - started}')
    source = original.replace_once(source, 'captured = {}',
        'captured = {}\n        _register_context(rows, captured, states)')
    source = original.replace_once(source, 'CODE / "interior_ssim_reuse.py", CODE / "evaluate_interior_ssim.py",',
        'CODE / "fused_ssim_formula.py", CODE / "ssim_output_census.py", CODE / "interior_ssim_reuse.py", CODE / "evaluate_interior_ssim.py",')
    return source


def run(fault=None):
    pair = Path(sys.argv[sys.argv.index("--pair")+1])
    pair.resolve().relative_to(OWN)
    os.environ["TRITON_CACHE_DIR"] = str(pair.parent / "triton_cache")
    source_path = pair.parent / "ACTUAL_MEASUREMENT_SOURCE.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("x", encoding="utf-8") as stream: stream.write(composed_main())
    metadata = entry.activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
    from ssim_output_census import new_trace
    output_trace, registry = new_trace(), []
    calls = []
    scientific_context = []

    def fault_before(ordinal):
        calls.append(ordinal)
        if ordinal == 5 and fault == "error":
            raise RuntimeError("owned_R134_V3_fifth_SSIM_failure")
        if ordinal == 5 and fault == "cancel":
            import signal
            os.kill(os.getpid(), signal.SIGINT)
            raise AssertionError("The real Linux SIGINT did not interrupt evaluation")

    namespace = dict(vars(original.entry))
    namespace.update(__file__=str(Path(__file__)), STARTED=STARTED,
        compose_view_major=original.compose_view_major, finish_reference_view=original.finish_reference_view,
        finish_coupled_view=original.finish_coupled_view, _register_inner=registry.append,
        _fault_before_SSIM=fault_before, _output_trace=output_trace)
    namespace["_register_context"] = lambda rows, captured, states: scientific_context.append((rows,captured,states))
    namespace["activate_reference"] = lambda runtime: metadata
    error = None
    with original.native_conv_work() as conv_trace:
        namespace["_conv_trace"] = conv_trace
        try:
            exec(compile(source_path.read_text(encoding="utf-8"), "<private-complete-fused-SSIM>", "exec"), namespace)
            namespace["main"]()
        except BaseException as exc: error = exc
        finally:
            for inner in registry: inner.clear()
    import evaluate_pair
    import torch
    torch.cuda.synchronize()
    checks = {"native_conv_binding_restored":conv_trace["original_binding_restored"],
        "private_helper_bindings_removed":not hasattr(evaluate_pair,"_observe_gaussian") and not hasattr(evaluate_pair,"_finish_view"),
        "all_owned_caches_and_last_maps_closed":bool(registry) and all(inner.receipt()["resident_accounted_bytes"] == 0
            and inner.receipt().get("retained_last_map_bytes",0) == 0 for inner in registry),
        "actual_convolution_work_nonzero":conv_trace["successful_CUDA_output_elements"] > 0,
        "owned_CUDA_under4GiB":torch.cuda.max_memory_reserved() <= 4*1024**3}
    payload = {"trace":conv_trace,"checks":checks,"fault":fault,
        "error":None if error is None else {"type":type(error).__name__,"message":str(error)},
        "complete_child_seconds":time.perf_counter()-STARTED,"new_optimization_count":0}
    atomic_json(pair.parent / "CONV_WORK.json", payload)
    atomic_json(pair.parent / "OUTPUT_WORK.json", {"trace":output_trace,"checks":checks,
        "fault":fault,"error":payload["error"],"attempted_ordinals":calls,
        "scientific_prefix_rows": [] if not scientific_context else scientific_context[0][0],
        "complete_original_parameters_before_fault": [] if not scientific_context else list(scientific_context[0][1].values()),
        "composed_source":data_identity(source_path),"original_mean_and_conv_unchanged":True,"new_optimization_count":0})
    if error is not None: raise error
    if not all(checks.values()): raise AssertionError("An original complete SSIM ownership or output-work protection failed")


if __name__ == "__main__":run()

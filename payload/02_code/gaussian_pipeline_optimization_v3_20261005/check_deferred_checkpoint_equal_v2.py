"""Repeat real CUDA semantics and exercise the pre-counter interrupt boundary."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("check_deferred_checkpoint_equal_001.py").read_text(encoding="utf-8")
    early = '''    raw_eq, raw_sync = torch.eq, torch.cuda.synchronize
    early_syncs = []
    early_left = torch.arange(64, device="cuda", dtype=torch.float32)
    early_right = early_left.clone()
    def interrupted_eq(left, right, **kwargs):
        raw_eq(left, right, **kwargs)
        raise KeyboardInterrupt("early_CUDA_before_result_counter")
    def measured_sync(*args, **kwargs):
        value = raw_sync(*args, **kwargs)
        early_syncs.append("actual_synchronize_returned")
        return value
    torch.eq, torch.cuda.synchronize = interrupted_eq, measured_sync
    early_error = None
    try:
        with deferred_checkpoint_equal(adapter, True) as early_trace:
            adapter.exact_equal([early_left], [early_right])
    except KeyboardInterrupt as error:
        early_error = str(error)
    finally:
        torch.eq, torch.cuda.synchronize = raw_eq, raw_sync
    early_checks = {
        "actual_early_interrupt_propagated": early_error == "early_CUDA_before_result_counter",
        "issued_before_counter_drained": early_trace["CUDA_work_may_be_issued"]
            and early_trace["CUDA_comparison_leaves"] == 0
            and early_trace["exception_owned_current_stream_drained"]
            and early_syncs == ["actual_synchronize_returned"],
        "all_original_early_bindings_restored": early_trace["original_comparator_and_tensor_equal_bindings_restored"]
            and torch.eq is raw_eq and torch.cuda.synchronize is raw_sync,
    }
    atomic_json(output / "EARLY_CUDA_INTERRUPT_REGRESSION.json", {"checks": early_checks,
        "trace": early_trace, "actual_synchronizations": early_syncs})
    if not all(early_checks.values()):
        raise AssertionError("The real early CUDA interruption was not safely drained")
'''
    replacements = [
        ('from deferred_checkpoint_equal_001 import deferred_checkpoint_equal',
         'from deferred_checkpoint_equal_v2 import deferred_checkpoint_equal'),
        ('ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/checkpoint_adapter.py", public_path)]',
         'ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/checkpoint_adapter.py", public_path, Path(__file__).with_name("deferred_checkpoint_equal_v2.py"), Path(__file__).with_name("check_deferred_checkpoint_equal_001.py")]'),
        ('    torch.cuda.synchronize()\n    checks = {', early + '    torch.cuda.synchronize()\n    checks = {'),
        ('    atomic_json(output / "RESULT.json", {"checks": checks,',
         '    checks.update(early_checks)\n    atomic_json(output / "RESULT.json", {"checks": checks,'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The frozen real CUDA semantic component changed")
        source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_real_CUDA_semantics_v2"}
    exec(compile(source, "<same_generic_public_and_actual_early_CUDA_regression>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

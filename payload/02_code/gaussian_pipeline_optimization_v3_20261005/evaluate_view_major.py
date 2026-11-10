"""Reuse the exact full-input evaluator guard with only consumer order changed."""

import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
import evaluate_interior_ssim
from view_major_evaluation import compose_view_major, finish_reference_view


def main():
    source = inspect.getsource(evaluate_interior_ssim.main)
    replacements = {
        'choices=("native", "interior")': 'choices=("native", "view")',
        'inner = ReferenceSSIM(loss_utils.create_window) if args.mode == "native" else InteriorSSIM(loss_utils.create_window)':
            'inner = ReferenceSSIM(loss_utils.create_window)\n        finished_views = []\n        def finish_view():\n            finish_reference_view(inner, finished_views)',
        'pair_source = inspect.getsource(evaluate_pair.main)':
            'pair_source = compose_view_major(evaluate_pair.main) if args.mode == "view" else inspect.getsource(evaluate_pair.main)',
        'pair_namespace["_observe_gaussian"] = observe':
            'pair_namespace["_observe_gaussian"] = observe\n        pair_namespace["_finish_view"] = finish_view',
        'CODE / "interior_ssim_reuse.py"': 'CODE / "view_major_evaluation.py", CODE / "evaluate_interior_ssim.py"',
        'evaluate_pair._observe_gaussian = observe':
            'evaluate_pair._observe_gaussian = observe\n            evaluate_pair._finish_view = finish_view',
        'del evaluate_pair._observe_gaussian':
            'del evaluate_pair._observe_gaussian\n            del evaluate_pair._finish_view',
        'and not hasattr(evaluate_pair, "_observe_gaussian") and private_filename not in linecache.cache':
            'and not hasattr(evaluate_pair, "_observe_gaussian") and not hasattr(evaluate_pair, "_finish_view") and private_filename not in linecache.cache',
        '"reference_stride": list(reference.stride()), "input_bits_and_versions_unchanged": unchanged,':
            '"reference_stride": list(reference.stride()), "reference_pointer": reference.untyped_storage().data_ptr(), "reference_offset": reference.storage_offset(), "reference_version": reference._version, "reference_dtype": str(reference.dtype), "reference_device": str(reference.device), "window_size": window_size, "input_bits_and_versions_unchanged": unchanged,',
        'if args.mode == "interior":\n            checks["all48_actual_target_crops_reused"] = cache_receipt["interior_reused"] == 48':
            'if args.mode == "view":\n            def reference_key(row):\n                return (row["reference_pointer"], row["reference_offset"], tuple(row["shape"]), tuple(row["reference_stride"]), row["reference_version"], row["reference_dtype"], row["reference_device"], row["window_size"])\n            expected_entries = [len({reference_key(row) for row in rows[index*4:(index+1)*4]}) for index in range(24)]\n            checks["all24_reference_views_released_after_both_models"] = len(finished_views) == 24 and all(row["released_reference_entries"] == expected for row, expected in zip(finished_views, expected_entries))',
        '"mode": args.mode, "sources": sources, "rows": rows, "model_states": states, "cache_before_clear": cache_receipt,':
            '"mode": args.mode, "sources": sources, "rows": rows, "model_states": states, "cache_before_clear": cache_receipt, "finished_reference_views": finished_views,'
    }
    for anchor, replacement in replacements.items():
        if source.count(anchor) != 1:
            raise ValueError("The exact original measurement source changed: " + anchor)
        source = source.replace(anchor, replacement)
    namespace = dict(vars(evaluate_interior_ssim))
    namespace.update(__file__=str(Path(__file__)), compose_view_major=compose_view_major, finish_reference_view=finish_reference_view)
    exec(compile(source, "<private-current-view-major-measurement>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

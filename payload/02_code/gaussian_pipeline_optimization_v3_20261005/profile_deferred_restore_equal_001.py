"""Compose the original complete restore with Boolean work reduction only."""

from pathlib import Path


class CandidateSource:
    def read_text(self, encoding):
        source = Path(__file__).with_name("profile_native_restore_cost.py").read_text(encoding=encoding)
        replacements = [
            ('from private_checkpoint_load import private_checkpoint_load',
             'from private_checkpoint_load import private_checkpoint_load\nfrom deferred_checkpoint_equal_001 import deferred_checkpoint_equal'),
            ('OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001"',
             'OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168"'),
            ('    parser.add_argument("--output", type=Path, required=True)',
             '    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--equality-mode", choices=("native", "deferred"), required=True)'),
            ('        RUNTIME / "source/gaussian-splatting/scene/__init__.py"]',
             '        RUNTIME / "source/gaussian-splatting/scene/__init__.py", CODE / "profile_cpu_dependency_restore.py", CODE / "deferred_checkpoint_equal_001.py"]'),
            ('with private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace:',
             'with deferred_checkpoint_equal(checkpoint_adapter, args.equality_mode == "deferred") as equality, private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace:'),
            ('            restored = read(output / "RESTORE_CHECK.json")',
             '            atomic_json(output / "EQUALITY_WORK_TRACE.json", equality)\n            restored = read(output / "RESTORE_CHECK.json")'),
            ('                "all_original_nine_restore_fields":',
             '                "original_equality_mechanism_closed": equality["original_comparator_and_tensor_equal_bindings_restored"],\n                "all_original_nine_restore_fields":'),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The frozen complete native restore source changed at: " + before)
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The original complete resource wrapper changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_deferred_boolean_restore"}
    exec(compile(source, "<original_CPU_restore_with_deferred_Boolean_equal>", "exec"), namespace)
    namespace["BASE"] = CandidateSource()
    namespace["main"]()


if __name__ == "__main__":
    main()

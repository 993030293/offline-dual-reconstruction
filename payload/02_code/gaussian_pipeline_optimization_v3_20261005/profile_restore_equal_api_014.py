"""Run one unchanged complete native restore with an equality-only observer."""

from pathlib import Path

class ObservedSource:
    def read_text(self, encoding):
        source = Path(__file__).with_name("profile_native_restore_cost.py").read_text(encoding=encoding)
        replacements = [
            ('from private_checkpoint_load import private_checkpoint_load',
             'from private_checkpoint_load import private_checkpoint_load\nfrom observed_restore_equal_014 import observe_restore_equality'),
            ('OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001"',
             'OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014"'),
            ('        RUNTIME / "source/gaussian-splatting/scene/__init__.py"]',
             '        RUNTIME / "source/gaussian-splatting/scene/__init__.py", CODE / "profile_cpu_dependency_restore.py", CODE / "observed_restore_equal_014.py"]'),
            ('with private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace:',
             'with observe_restore_equality(torch) as equality, private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace:'),
            ('            restored = read(output / "RESTORE_CHECK.json")',
             '            atomic_json(output / "EQUALITY_API_TRACE.json", equality)\n            restored = read(output / "RESTORE_CHECK.json")'),
            ('                "all_original_nine_restore_fields":',
             '                "original_equality_observer_closed": equality["original_torch_equal_binding_restored"] and equality["profiler_context_closed"],\n                "all_original_nine_restore_fields":'),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The exact original restore source changed at: " + before)
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The complete original native resource source changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_native_restore_equality_observer"}
    exec(compile(source, "<original_CPU_restore_with_registered180second_observer>", "exec"), namespace)
    namespace["BASE"] = ObservedSource()
    namespace["main"]()


if __name__ == "__main__":
    main()

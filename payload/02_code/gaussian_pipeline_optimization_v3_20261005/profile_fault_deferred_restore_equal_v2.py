"""Retain the original fault entry while repairing the private injection only."""

from pathlib import Path

from profile_fault_deferred_restore_equal_001 import FaultSource


class FaultSourceV2(FaultSource):
    def read_text(self, encoding):
        source = super().read_text(encoding)
        replacements = [
            ('from fault_deferred_restore_hook_001 import actual_restore_fault',
             'from fault_deferred_restore_hook_v2 import actual_restore_fault'),
            ('CODE / "fault_deferred_restore_hook_001.py"]',
             'CODE / "fault_deferred_restore_hook_001.py", CODE / "fault_deferred_restore_hook_v2.py", CODE / "profile_fault_deferred_restore_equal_001.py"]'),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The retained actual fault source changed")
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The original resource source changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_actual_fault_v2"}
    exec(compile(source, "<original_complete_restore_actual_fault_v2>", "exec"), namespace)
    namespace["BASE"] = FaultSourceV2()
    namespace["main"]()


if __name__ == "__main__":
    main()

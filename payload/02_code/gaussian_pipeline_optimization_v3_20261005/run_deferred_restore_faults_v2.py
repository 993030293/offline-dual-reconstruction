"""Preserve the first failed controller and retry only the repaired injection."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("run_deferred_restore_faults_001.py").read_text(encoding="utf-8")
    replacements = [
        ('linux(CODE / "profile_fault_deferred_restore_equal_001.py")',
         'linux(CODE / "profile_fault_deferred_restore_equal_v2.py")'),
        ('linux(CODE / "stop_owned_restore_fault_linux_001.py")',
         'linux(CODE / "stop_owned_restore_fault_linux_v2.py")'),
        ('for name in (Path(__file__).name,',
         'for name in (Path(__file__).name, "run_deferred_restore_faults_001.py", "profile_fault_deferred_restore_equal_v2.py", "fault_deferred_restore_hook_v2.py", "stop_owned_restore_fault_linux_v2.py",'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The retained actual fault controller source changed")
        source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_actual_fault_controller_v2"}
    exec(compile(source, "<retained_actual_fault_controller_v2>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

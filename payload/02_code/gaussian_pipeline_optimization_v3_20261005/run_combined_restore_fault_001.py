"""Verify both real candidate faults with one complete Scene and original checks."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("run_deferred_restore_faults_001.py").read_text(encoding="utf-8")
    replacements = [
        ('linux(CODE / "profile_fault_deferred_restore_equal_001.py")',
         'linux(CODE / "profile_combined_restore_fault_v3.py")'),
        ('linux(CODE / "stop_owned_restore_fault_linux_001.py")',
         'linux(CODE / "stop_owned_combined_restore_fault_linux_001.py")'),
        ('for name in (Path(__file__).name,',
         'for name in (Path(__file__).name, "run_deferred_restore_faults_001.py", "profile_combined_restore_fault_v3.py", "profile_fault_deferred_restore_equal_v2.py", "fault_deferred_restore_hook_v2.py", "deferred_checkpoint_equal_v2.py", "stop_owned_combined_restore_fault_linux_001.py",'),
        ('for kind in ("mismatch", "cancel"):', 'for kind in ("cancel",):'),
        ('for mode in ("native", "deferred"):', 'for mode in ("deferred",):'),
        ('                row = {"kind": kind, "mode": mode, "checks": checks,',
         '                checks["actual_first_prepare_mismatch_then_independent_SIGINT"] = all(read(folder / "first_mismatch/MISMATCH_PATH.json")["checks"].values())\n                row = {"kind": kind, "mode": mode, "checks": checks,'),
        ('"all_four_actual_fault_paths": len(rows) == 4', '"both_actual_mismatch_and_SIGINT_candidate_paths": len(rows) == 1'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The retained actual fault source or original checks changed")
        source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_combined_candidate_faults"}
    exec(compile(source, "<same_original_fault_guard_for_combined_candidate>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

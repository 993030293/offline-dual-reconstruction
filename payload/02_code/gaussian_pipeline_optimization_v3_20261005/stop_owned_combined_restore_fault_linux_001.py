"""Drain only the exact combined fault process with saved birth and argv."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("stop_owned_restore_linux_001.py").read_text(encoding="utf-8")
    before = 'with_name("profile_deferred_restore_equal_001.py")'
    if source.count(before) != 1:
        raise ValueError("The saved native process drain source changed")
    source = source.replace(before, 'with_name("profile_combined_restore_fault_v3.py")')
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_exact_combined_fault_drain"}
    exec(compile(source, "<birth_checked_combined_native_fault_drain>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

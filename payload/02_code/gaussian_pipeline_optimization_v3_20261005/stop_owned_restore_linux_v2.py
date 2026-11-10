"""Drain only the exact complete second equality entry by saved birth and argv."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("stop_owned_restore_linux_001.py").read_text(encoding="utf-8")
    before = 'with_name("profile_deferred_restore_equal_001.py")'
    if source.count(before) != 1:
        raise ValueError("The frozen owned Linux drain changed")
    source = source.replace(before, 'with_name("profile_deferred_restore_equal_v2.py")')
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_exact_equality_v2_drain"}
    exec(compile(source, "<birth_checked_complete_equality_v2_drain>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

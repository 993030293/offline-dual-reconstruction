"""Preserve the failed wrapper and expose its required subprocess constant."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("run_deferred_restore_equal_pair_v2.py").read_text(encoding="utf-8")
    replacements = [
        ('    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW',
         '    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW\n    STDOUT = subprocess.STDOUT'),
        ('after = \'(Path(__file__).name, "run_deferred_restore_equal_pair_001.py",',
         'after = \'(Path(__file__).name, "run_deferred_restore_equal_pair_v2.py", "run_deferred_restore_equal_pair_001.py",'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The retained failed complete-pair wrapper changed")
        source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_pair_constant_repair"}
    exec(compile(source, "<complete_pair_with_required_subprocess_constant>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

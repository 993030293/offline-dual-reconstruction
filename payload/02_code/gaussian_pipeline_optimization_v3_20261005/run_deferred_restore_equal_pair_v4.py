"""Run the repaired equality through the same complete side and pair oracle."""

from pathlib import Path


def main():
    code = Path(__file__).parent
    owner_source = (code / "run_deferred_restore_equal_pair_v2.py").read_text(encoding="utf-8")
    for before, after in (
        ('    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW',
         '    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW\n    STDOUT = subprocess.STDOUT'),
        ('/stop_owned_restore_linux_001.py", folder]', '/stop_owned_restore_linux_v2.py", folder]'),
    ):
        if owner_source.count(before) != 1:
            raise ValueError("The saved process owner source changed")
        owner_source = owner_source.replace(before, after)
    owner_namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_pair_v4_processes"}
    exec(compile(owner_source, "<saved_birth_checked_owner_with_v2_native_argv>", "exec"), owner_namespace)
    source = (code / "run_deferred_restore_equal_pair_001.py").read_text(encoding="utf-8")
    replacements = [
        ('linux(CODE / "profile_deferred_restore_equal_001.py")',
         'linux(CODE / "profile_deferred_restore_equal_v2.py")'),
        ('for name in (Path(__file__).name, "deferred_checkpoint_equal_001.py",',
         'for name in (Path(__file__).name, "run_deferred_restore_equal_pair_001.py", "run_deferred_restore_equal_pair_v2.py", "deferred_checkpoint_equal_v2.py", "profile_deferred_restore_equal_v2.py", "stop_owned_restore_linux_v2.py", "stop_owned_restore_linux_001.py", "deferred_checkpoint_equal_001.py",'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The original full pair or input oracle changed")
        source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_equality_v2_pair"}
    exec(compile(source, "<same_original_complete_pair_for_equality_v2>", "exec"), namespace)
    owner = owner_namespace["OwnedProcesses"]()
    namespace["subprocess"] = owner
    try:
        namespace["main"]()
    except BaseException:
        owner.close_failed()
        raise


if __name__ == "__main__":
    main()

"""Keep normal complete restore unchanged and repair early CUDA cleanup only."""

from pathlib import Path

from profile_deferred_restore_equal_001 import CandidateSource


class CandidateSourceV2(CandidateSource):
    def read_text(self, encoding):
        source = super().read_text(encoding)
        replacements = [
            ('from deferred_checkpoint_equal_001 import deferred_checkpoint_equal',
             'from deferred_checkpoint_equal_v2 import deferred_checkpoint_equal'),
            ('CODE / "deferred_checkpoint_equal_001.py"]',
             'CODE / "deferred_checkpoint_equal_001.py", CODE / "deferred_checkpoint_equal_v2.py", CODE / "profile_deferred_restore_equal_001.py"]'),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The frozen original complete equality source changed")
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The original complete resource entry changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_restore_equality_v2"}
    exec(compile(source, "<same_original_complete_restore_with_early_CUDA_mark>", "exec"), namespace)
    namespace["BASE"] = CandidateSourceV2()
    namespace["main"]()


if __name__ == "__main__":
    main()

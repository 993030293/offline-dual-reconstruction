"""Keep the complete original restore while injecting one scoped real fault."""

from pathlib import Path

from profile_deferred_restore_equal_001 import CandidateSource


class FaultSource(CandidateSource):
    def read_text(self, encoding):
        source = super().read_text(encoding)
        before = "with deferred_checkpoint_equal(checkpoint_adapter, args.equality_mode == \"deferred\") as equality, private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace:"
        replacements = [
            ("from deferred_checkpoint_equal_001 import deferred_checkpoint_equal",
             "from deferred_checkpoint_equal_001 import deferred_checkpoint_equal\nfrom fault_deferred_restore_hook_001 import actual_restore_fault"),
            ('    parser.add_argument("--equality-mode", choices=("native", "deferred"), required=True)',
             '    parser.add_argument("--equality-mode", choices=("native", "deferred"), required=True)\n    parser.add_argument("--fault", choices=("mismatch", "cancel"), required=True)'),
            ('CODE / "deferred_checkpoint_equal_001.py"]',
             'CODE / "deferred_checkpoint_equal_001.py", CODE / "profile_deferred_restore_equal_001.py", CODE / "fault_deferred_restore_hook_001.py"]'),
            (before, before[:-1] + ", actual_restore_fault(checkpoint_adapter, torch, output, args.fault) as injected:"),
            ('    except BaseException as error:\n        atomic_json(output / "FAILURE.json",',
             '    except BaseException as error:\n        if "equality" in locals():\n            atomic_json(output / "EQUALITY_EXIT_TRACE.json", equality)\n        atomic_json(output / "FAILURE.json",'),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The exact original fault entry source changed at: " + before)
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The original complete CPU resource source changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_restore_actual_fault"}
    exec(compile(source, "<original_complete_restore_actual_fault>", "exec"), namespace)
    namespace["BASE"] = FaultSource()
    namespace["main"]()


if __name__ == "__main__":
    main()

"""Measure one original complete restore seed creation without skipping work."""

from pathlib import Path

from profile_deferred_restore_equal_v2 import CandidateSourceV2


class SeedSource(CandidateSourceV2):
    def read_text(self, encoding):
        source = super().read_text(encoding)
        replacements = [
            ('from deferred_checkpoint_equal_v2 import deferred_checkpoint_equal',
             'from deferred_checkpoint_equal_v2 import deferred_checkpoint_equal\nfrom observe_seed_initialization_015 import observe_seed_initialization'),
            ('OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168"',
             'OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup015"'),
            ('CODE / "profile_deferred_restore_equal_001.py"]',
             'CODE / "profile_deferred_restore_equal_001.py", CODE / "profile_deferred_restore_equal_v2.py", CODE / "observe_seed_initialization_015.py"]'),
            ('    args = parser.parse_args()',
             '    args = parser.parse_args()\n    if args.equality_mode != "deferred":\n        raise ValueError("The current original seed diagnostic keeps accepted V2 equality")'),
            ('            with (camera_loader_adapter(DATASET), camera_execution(**{key: flags[key]',
             '            with (observe_seed_initialization(GaussianModel, torch) as seed, camera_loader_adapter(DATASET), camera_execution(**{key: flags[key]'),
            ('            phases["complete_original_Scene_seconds"] = time.perf_counter() - before',
             '            phases["complete_original_Scene_seconds"] = time.perf_counter() - before\n            atomic_json(output / "SEED_INITIALIZATION_TRACE.json", seed)'),
            ('                "original_equality_mechanism_closed":',
             '                "original_seed_method_observer_closed": seed["original_create_from_pcd_binding_restored"] and len(seed["calls"]) == 1,\n                "original_equality_mechanism_closed":'),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The current complete seed diagnostic source changed")
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The current complete resource source changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_original_seed_cost"}
    exec(compile(source, "<same_current_original_restore_with_seed_observer>", "exec"), namespace)
    namespace["BASE"] = SeedSource()
    namespace["main"]()


if __name__ == "__main__":
    main()

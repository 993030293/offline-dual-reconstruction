"""Exercise real prepare mismatch and early SIGINT on one complete private Scene."""

from pathlib import Path

from profile_fault_deferred_restore_equal_v2 import FaultSourceV2


class CombinedFaultSource(FaultSourceV2):
    def read_text(self, encoding):
        source = super().read_text(encoding)
        scope = '            with deferred_checkpoint_equal(checkpoint_adapter, args.equality_mode == "deferred") as equality, private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace, actual_restore_fault(checkpoint_adapter, torch, output, args.fault) as injected:'
        mismatch = '''            mismatch_output = output / "first_mismatch"
            mismatch_output.mkdir()
            mismatch_error = None
            try:
                with deferred_checkpoint_equal(checkpoint_adapter, True) as mismatch_equal, private_checkpoint_load(owned) as mismatch_loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as mismatch_recovery, actual_restore_fault(checkpoint_adapter, torch, mismatch_output, "mismatch") as mismatch_injected:
                    restore.prepare(scene.gaussians, opt_group.extract(arguments), scene)
            except AssertionError as error:
                mismatch_error = str(error)
            mismatch_fault = read(mismatch_output / "ACTUAL_FAULT_TRACE.json")
            mismatch_checks = {
                "actual_original_prepare_AssertionError": mismatch_error == "Restored tensors or optimizer state differ from the saved state"
                    and mismatch_fault["actual_exception_type"] == "AssertionError",
                "actual_second_private_state_value_changed": mismatch_fault["private_second_state_mutated"]
                    and mismatch_fault["comparison_scope_live"] and mismatch_fault["actual_CUDA_compare_API_calls"] > 0,
                "two_actual_private_independent_loads": len(mismatch_loads["calls"]) == 2
                    and mismatch_recovery["two_loaded_states_own_independent_nonempty_storages"]
                    and mismatch_recovery["two_complete_load_layouts_devices_and_alias_groups_exact"],
                "all_original_scope_bindings_and_stream_drained": mismatch_equal["original_comparator_and_tensor_equal_bindings_restored"]
                    and mismatch_equal["exception_owned_current_stream_drained"]
                    and mismatch_loads["original_global_loader_restored"]
                    and mismatch_recovery["all_bindings_restored"] and mismatch_recovery["recursive_depth_closed"]
                    and mismatch_fault["original_prepare_and_tensor_API_bindings_restored"] and mismatch_fault["owned_signal_thread_closed"],
                "failed_prepare_did_not_publish_success_or_resumed_state": restore.restored is None
                    and not (output / "RESTORE_CHECK.json").exists() and not (output / "RESULT.json").exists(),
            }
            atomic_json(mismatch_output / "MISMATCH_PATH.json", {"checks": mismatch_checks,
                "actual_original_exception_message": mismatch_error, "equality": mismatch_equal,
                "private_loads": mismatch_loads, "recovery": mismatch_recovery, "fault": mismatch_fault})
            if not all(mismatch_checks.values()):
                raise RuntimeError("The actual first prepare mismatch did not safely close")
'''
        replacements = [
            ('from deferred_checkpoint_equal_001 import deferred_checkpoint_equal',
             'from deferred_checkpoint_equal_v2 import deferred_checkpoint_equal'),
            ('CODE / "profile_fault_deferred_restore_equal_001.py"]',
             'CODE / "profile_fault_deferred_restore_equal_001.py", CODE / "profile_fault_deferred_restore_equal_v2.py", CODE / "deferred_checkpoint_equal_v2.py"]'),
            ('    args = parser.parse_args()',
             '    args = parser.parse_args()\n    if args.equality_mode != "deferred" or args.fault != "cancel":\n        raise ValueError("The registered combined entry is candidate mismatch then SIGINT only")'),
            (scope, mismatch + scope),
        ]
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("The frozen complete fault entry changed")
            source = source.replace(before, after)
        return source


def main():
    source = Path(__file__).with_name("profile_cpu_dependency_restore.py").read_text(encoding="utf-8")
    before = "maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED))"
    if source.count(before) != 1:
        raise ValueError("The frozen original complete resource entry changed")
    source = source.replace(before, "maximum_seconds=max(1, 180 - (time.perf_counter() - STARTED))")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_combined_prepare_faults"}
    exec(compile(source, "<original_complete_restore_with_two_real_prepare_faults>", "exec"), namespace)
    namespace["BASE"] = CombinedFaultSource()
    namespace["main"]()


if __name__ == "__main__":
    main()

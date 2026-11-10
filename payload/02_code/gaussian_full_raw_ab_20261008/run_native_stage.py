"""Bind a dynamic recording to preserved native metric and export stages."""

import argparse
from contextlib import ExitStack
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from gpu_lease import gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("recovery", "temporal", "roles", "metric", "export"), required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allow-pure-sim3-recovery", action="store_true")
    parser.add_argument("--temporal-matching-seconds", type=int, default=1800)
    parser.add_argument("--export-qa-mode", choices=("serial", "dag"))
    args, remaining = parser.parse_known_args()
    module_name = {"recovery": "run_pair_recovery", "temporal": "run_temporal_augmentation", "roles": "audit_camera_roles",
                   "metric": "run_metric_refinement", "export": "export_admitted_map"}[args.kind]
    native = importlib.import_module(module_name)
    output = Path(remaining[remaining.index("--output") + 1])
    wrapper = output.parent / (output.name + "_wrapper.json")
    if wrapper.exists() or output.exists():
        raise FileExistsError("A native recording stage requires unique outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    identities = [data_identity(Path(__file__)), data_identity(Path(native.__file__))]
    atomic_json(wrapper, {"status": "frozen_before_native_stage", "kind": args.kind,
        "mode": args.mode, "raw": str(args.raw), "sources": identities, "arguments": remaining,
        "temporal_matching_budget_revision": {"original_seconds": 600, "requested_seconds": args.temporal_matching_seconds,
            "scope": "temporal matches_importer only", "quality_geometry_parameters_unchanged": True},
        "all_original_quality_and_geometry_thresholds_preserved": True})
    started = time.perf_counter()
    trace = {}
    import initial_selection
    old_raw = initial_selection.RAW
    initial_selection.RAW = args.raw
    old_argv = sys.argv
    original_main = native.main
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    gpu_calls = [0]

    def guarded_run(command, *parameters, **keywords):
        names = list(map(str, command))
        if args.kind == "temporal" and len(names) > 1 and names[1] == "matches_importer":
            if keywords.get("timeout") != 600 or not 600 <= args.temporal_matching_seconds <= 1800:
                raise ValueError("The explicit temporal matching budget contract changed")
            keywords["timeout"] = args.temporal_matching_seconds
            trace["temporal_matching_budget_revision"] = {"original_seconds": 600, "actual_seconds": args.temporal_matching_seconds}
        if len(names) > 1 and names[1] in ("feature_extractor", "matches_importer"):
            gpu_calls[0] += 1
            receipt = output.parent / (output.name + "_GPU_LEASE_" + str(gpu_calls[0]) + ".json")
            with gpu_lease(receipt, output.name + "_" + names[1]):
                return original_subprocess_run(command, *parameters, **keywords)
        return original_subprocess_run(command, *parameters, **keywords)

    def scoped_artifact_worker(command, *parameters, **keywords):
        values = list(map(str, command))
        original_worker = V3 / "export_qa_artifact_worker.py"
        if args.export_qa_mode is not None and str(original_worker) in values:
            values[values.index(str(original_worker))] = str(Path(__file__).with_name("export_worker_raw_scope.py"))
            trace["R140_worker_campaign_scope_binding"] = {
                "original": data_identity(original_worker),
                "scope_adapter": data_identity(Path(__file__).with_name("export_worker_raw_scope.py")),
                "all_original_worker_operations_and_parameter_values_unchanged": True}
            return original_subprocess_popen(values, *parameters, **keywords)
        return original_subprocess_popen(command, *parameters, **keywords)
    try:
        subprocess.run = guarded_run
        if args.export_qa_mode is not None:
            subprocess.Popen = scoped_artifact_worker
        with ExitStack() as stack:
            if args.mode == "optimized" and args.kind in ("roles", "export"):
                package = ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
                sys.path.insert(0, str(package / "code/validated_fresh_runtime"))
                import metric_admission
                from grid_support import grid_support_execution
                trace["R112"] = stack.enter_context(grid_support_execution(metric_admission, True))
            if args.mode == "optimized" and args.kind == "metric":
                from native_observation_checks import projection_validity_native, observation_identity_native
                native.projection_validity = projection_validity_native
                native.observation_identity = observation_identity_native
                trace["Q49"] = {"actual_native_observation_accessors_bound": True}
            if args.kind == "export":
                if args.allow_pure_sim3_recovery:
                    ba_path = Path(remaining[remaining.index("--ba") + 1]) / "RESULT.json"
                    ba = json.loads(ba_path.read_text())
                    detail = ba.get("results", {}).get("pure_sim3_alignment", {})
                    if (ba.get("status") != "pure_sim3_metric_interface_verified_BA_not_accepted"
                            or ba.get("selected") != "pure_sim3_alignment" or not ba.get("BA_failure_preserved")
                            or ba.get("original_BA_guard_passed") is not False or detail.get("active_priors") != 0
                            or not detail.get("common_observations_preserved") or not detail.get("projection", {}).get("valid")
                            or detail.get("reprojection_max_absolute_delta_px", float("inf")) > 1e-6):
                        raise ValueError("The explicit pure-similarity recovery has no complete valid oracle")
                source = inspect.getsource(original_main)
                changes = [
                    ('ba["status"] != "metric_interface_verified" or not ba["development_guard_passed"]\n                or ba["selected"] != "weak_position_prior" or ba["common_observation_p95_delta_px"] > .05',
                     'ba["status"] != "metric_interface_verified" or ba["selected"] not in ("weak_position_prior", "visual_metric_aligned")\n                or (ba["selected"] == "weak_position_prior" and (not ba["development_guard_passed"] or ba["common_observation_p95_delta_px"] > .05))'),
                    ('ba["results"]["weak_position_prior"]["model"]', 'ba["results"][ba["selected"]]["model"]'),
                ]
                for before, after in changes:
                    if source.count(before) != 1:
                        raise ValueError("The explicit native metric fallback anchors changed")
                    source = source.replace(before, after)
                if args.allow_pure_sim3_recovery:
                    status_anchor = 'ba["status"] != "metric_interface_verified" or ba["selected"] not in ("weak_position_prior", "visual_metric_aligned")'
                    if source.count(status_anchor) != 1:
                        raise ValueError("The explicit export status boundary changed")
                    source = source.replace(status_anchor,
                        'ba["status"] != "pure_sim3_metric_interface_verified_BA_not_accepted" or ba["selected"] != "pure_sim3_alignment"')
                namespace = dict(original_main.__globals__)
                import linecache
                source_name = "<explicit_native_selected_metric_export>"
                linecache.cache[source_name] = (len(source), None, source.splitlines(True), source_name)
                exec(compile(source, source_name, "exec"), namespace)
                native.main = namespace["main"]
                trace["metric_fallback"] = {"visual_metric_aligned_model_allowed_if_native_guard_selects_it": True,
                    "weak_prior_failure_preserved": True, "support_and_point_thresholds_unchanged": True,
                    "pure_sim3_recovery_explicitly_requested": args.allow_pure_sim3_recovery}
                if args.export_qa_mode is not None:
                    from segmented_export_execution import segmented_export_execution
                    from export_qa_dag import export_qa_dag_execution
                    trace["R132_common_both_sides"] = stack.enter_context(segmented_export_execution(
                        native, output.parent / (output.name + "_support"), True))
                    trace["R140"] = stack.enter_context(export_qa_dag_execution(
                        native, source, args.export_qa_mode, output.parent / (output.name + "_worker")))
            sys.argv = [str(Path(native.__file__)), *remaining]
            native.main()
        atomic_json(wrapper, {"status": "native_stage_completed", "kind": args.kind, "mode": args.mode,
            "raw": str(args.raw), "sources": identities, "arguments": remaining,
            "module_trace": trace, "elapsed_seconds": time.perf_counter() - started,
            "all_original_quality_and_geometry_thresholds_preserved": True})
    finally:
        initial_selection.RAW = old_raw
        native.main = original_main
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        sys.argv = old_argv


if __name__ == "__main__":
    main()

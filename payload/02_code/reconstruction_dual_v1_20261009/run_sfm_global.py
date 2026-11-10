"""Use the tested native global mapper with unchanged fresh feature matching."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback
import pycolmap

from common import OLD, atomic_json, read_json


def global_command(command):
    values = list(map(str, command))
    if len(values) < 2 or values[1] != "mapper":
        return values
    required = ("--database_path", "--image_path", "--output_path")
    if any(values.count(key) != 1 for key in required):
        raise ValueError("The native mapping input contract changed")
    result = [values[0], "global_mapper"]
    for key in required:
        result.extend([key, values[values.index(key) + 1]])
    result.extend(["--default_random_seed", "0", "--GlobalMapper.num_threads", "8",
        "--GlobalMapper.min_num_matches", str(pycolmap.GlobalPipelineOptions().min_num_matches),
        "--GlobalMapper.random_seed", "0", "--GlobalMapper.ba_refine_focal_length", "0",
        "--GlobalMapper.ba_refine_principal_point", "0", "--GlobalMapper.ba_refine_extra_params", "0",
        "--GlobalMapper.refine_sensor_from_rig", "0", "--GlobalMapper.gp_use_gpu", "0",
        "--GlobalMapper.ba_ceres_use_gpu", "0"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("optimized",), required=True)
    parser.add_argument("--mapper-seconds", type=int, default=1800)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("preserved_fresh_sfm", OLD / "run_sfm.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original = subprocess.Popen
    mapping_commands = []

    def launch(command, *positional, **keywords):
        values = global_command(command)
        if values != list(map(str, command)):
            mapping_commands.append(values)
            atomic_json(args.output / "BACKEND_SELECTION.json", {
                "backend": "COLMAP_native_global_CPU", "commands": mapping_commands,
                "coarse_poses_are_hints_only": True, "fixed_calibration": True,
                "GPU_solver_requested": False, "external_process_budget_seconds": args.mapper_seconds,
                "original_matching_and_feature_parameters_preserved": True})
        return original(values, *positional, **keywords)

    try:
        subprocess.Popen = launch
        native.run(args)
        selection_path = args.initial / 'INITIAL_TRACKABILITY_RECEIPT.json'
        selection = read_json(selection_path) if selection_path.is_file() else {}
        if selection.get('coarse_baseline_pair_extension_used_for_quality_and_final_SfM_graph'):
            contract_path = args.output / 'CONTRACT.json'
            contract = read_json(contract_path)
            contract.update(baseline_uses_same_candidate_graph_as_optimized=False,
                initial_candidate_graph_protocol=selection['protocol'],
                same_attempt_frozen_baseline_diverse_pair_file_consumed=True,
                original_matching_and_feature_options_preserved=True)
            atomic_json(contract_path, contract)
        if len(mapping_commands) != 1:
            raise ValueError("Expected exactly one global mapping stage")
        result = read_json(args.output / "RESULT.json")
        result["initial_solver"] = "COLMAP_native_global_CPU"
        result["actual_mapping_commands"] = mapping_commands
        atomic_json(args.output / "RESULT.json", result)
        atomic_json(args.output / "guided/RESULT.json", result)
        state = read_json(args.output / "STATE.json")
        for stage in state["stages"]:
            if stage["name"] == "guided_mapping":
                stage["commands"] = mapping_commands
        atomic_json(args.output / "STATE.json", state)
        print(json.dumps({"status": result["status"], "registered": result["best_registered_count"]}), flush=True)
    except BaseException:
        if args.output.exists():
            atomic_json(args.output / "FAILURE.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        subprocess.Popen = original


if __name__ == "__main__":
    main()

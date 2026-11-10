"""Compare complete original admitted-map exports with one sparse oracle edit."""

import time

IMPORT_STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V1OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
import export_admitted_map as native
from runtime import atomic_json, data_identity
from sparse_ply_oracle import sparse_ply_execution
from machine_json import exact_json_values


def normalize_json(value, folder, top=True):
    if isinstance(value, dict):
        observed = {"elapsed_seconds", "peak_RSS_bytes"} if top else set()
        return {key: normalize_json(item, folder, False) for key, item in value.items()
                if key not in observed}
    if isinstance(value, list):
        return [normalize_json(item, folder, False) for item in value]
    if isinstance(value, str):
        return value.replace(str(folder), "<CURRENT_EXPORT>").replace(
            folder.as_posix(), "<CURRENT_EXPORT>")
    return value


def compare_outputs(left, right):
    paths = [{p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
             for folder in (left, right)]
    if paths[0].keys() != paths[1].keys():
        raise AssertionError("Complete exported file membership differs")
    bytes_checked, files, json_files = 0, 0, 0
    for name in sorted(paths[0]):
        first, second = paths[0][name], paths[1][name]
        if first.suffix.lower() == ".json":
            a = normalize_json(json.loads(first.read_text(encoding="utf-8")), left)
            b = normalize_json(json.loads(second.read_text(encoding="utf-8")), right)
            try:
                exact_json_values(a, b)
            except AssertionError as error:
                raise AssertionError("Complete scientific JSON differs: "+name) from error
            json_files += 1
        else:
            if first.stat().st_size != second.stat().st_size:
                raise AssertionError("Complete export sizes differ: "+name)
            with first.open("rb") as a, second.open("rb") as b:
                while block := a.read(4*1024**2):
                    if block != b.read(len(block)):
                        raise AssertionError("Complete output bytes differ: "+name)
                    bytes_checked += len(block)
                if b.read(1):
                    raise AssertionError("Unexpected trailing export bytes")
            files += 1
    return {"complete_byte_pairs": files, "complete_JSON_pairs": json_files,
            "complete_compared_bytes": bytes_checked,
            "observed_JSON_top_level_omissions": ["elapsed_seconds", "peak_RSS_bytes"]}


def run_leaf(args):
    folder = args.output.resolve()
    folder.relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    identities = contract["sources_and_inputs"]
    paths = [Path(row["path"]) for row in identities]
    if [data_identity(path) for path in paths] != identities:
        raise RuntimeError("Frozen source or export input differs before the real child entry")
    leaf_prepare = time.perf_counter()-IMPORT_STARTED
    ba = V1OUT/"new_map_metric_BA_001"
    roles = V1OUT/"new_map_roles_postBA_001/TRAINING_ROLES.json"
    images = V1OUT/"e2_geometry_support_001/images"
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles),
                    "--images", str(images), "--output", str(folder)]
        with sparse_ply_execution(native, args.leaf_mode == "columns") as trace:
            native.main()
    finally:
        sys.argv = old_argv
    final_started = time.perf_counter()
    if [data_identity(path) for path in paths] != identities:
        raise RuntimeError("Frozen source or export input changed during the real child entry")
    if not trace["original_binding_restored"] or len(trace["calls"]) != 1:
        raise AssertionError("Original sparse oracle binding or call count differs")
    if trace["calls"][0]["status"] != "passed_complete_official_roundtrip":
        raise AssertionError("The complete official sparse PLY oracle did not pass")
    if sum(path.stat().st_size for path in folder.parent.rglob("*") if path.is_file()) > 30*1024**3:
        raise MemoryError("Actual owned exports exceed the original output budget")
    if shutil.disk_usage(folder).free < 20*1024**3:
        raise OSError("Original disk headroom violated")
    final_checks = time.perf_counter()-final_started
    result = {"status": "complete_original_export_and_protections_passed",
              "leaf_mode": args.leaf_mode, "folder": str(folder), "trace": trace,
              "phase_seconds": sum(row["complete_call_seconds"] for row in trace["calls"])
                  +trace["setup_seconds"]+trace["restore_seconds"]+leaf_prepare+final_checks,
              "complete_leaf_before_receipt_seconds": time.perf_counter()-IMPORT_STARTED,
              "counted": False}
    atomic_json(folder.parent/f"{folder.name}_leaf_RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf-mode", choices=("native", "columns"))
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    if args.leaf_mode is not None:
        if args.contract is None:
            raise ValueError("Each real leaf requires its frozen source/input contract")
        run_leaf(args)
        return
    output = args.output.resolve()
    output.relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005")
    output.mkdir(parents=True, exist_ok=False)
    batch_started = IMPORT_STARTED
    ba = V1OUT/"new_map_metric_BA_001"
    roles = V1OUT/"new_map_roles_postBA_001/TRAINING_ROLES.json"
    images = V1OUT/"e2_geometry_support_001/images"
    sources = [Path(__file__), Path(__file__).with_name("sparse_ply_oracle.py"),
               Path(__file__).with_name("machine_json.py"),
               Path(native.__file__), V1/"runtime.py", V1/"run_metric_refinement.py"]
    helpers = ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"
    sources.append(helpers/"common.py")
    sys.path.insert(0, str(helpers))
    import common
    sources += [common.PACKAGE/"code/validated_fresh_runtime/metric_admission.py",
                common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = [roles, ba/"RESULT.json"]+sorted((ba/"weak_position_prior/model").glob("*.bin"))
    inputs += sorted(p for p in images.iterdir() if p.is_file())
    identities = [data_identity(p) for p in sources+inputs]
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshot/f"{index:02d}_{path.name}")
    contract_path = output/"CONTRACT.json"
    atomic_json(contract_path, {"sources_and_inputs": identities,
        "mechanism": "R111_official_sparse_PLY_roundtrip_column_assembly", "pairs": 3,
        "original_main_and_all_original_parameters": True, "first_preparation_divisor": 6,
        "each_side_actual_fresh_python_child_with_complete_process_join": True,
        "phase_charges_child_receipt_publication_and_process_close": True,
        "minimum_phase_reduction": .1, "maximum_each_whole_ratio": 1.1,
        "whole_median_nonregression_required": True, "counted": False})
    first_prepare = time.perf_counter()-batch_started
    pairs = []
    try:
        for trial in range(3):
            rows = {}
            for label in (("native", "columns") if trial%2 == 0 else ("columns", "native")):
                folder = output/f"pair{trial}_{label.ljust(7, '_')}"
                started = time.perf_counter()
                with (output/f"{folder.name}.log").open("xb") as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--output", str(folder), "--leaf-mode", label,
                        "--contract", str(contract_path)], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                if process.returncode != 0:
                    raise RuntimeError(f"Real original export child failed: {label}, {process.returncode}")
                leaf = json.loads((output/f"{folder.name}_leaf_RESULT.json").read_text(encoding="utf-8"))
                if leaf["status"] != "complete_original_export_and_protections_passed":
                    raise AssertionError("The source-bound complete child result is not passed")
                whole = time.perf_counter()-started
                external = whole-leaf["complete_leaf_before_receipt_seconds"]
                if external < 0:
                    raise AssertionError("Parent process envelope is smaller than actual child execution")
                rows[label] = {"whole_seconds": whole+first_prepare/6,
                    "phase_seconds": leaf["phase_seconds"]+external+first_prepare/6,
                    "child_start_receipt_read_publication_and_process_close_seconds": external,
                    "child_returncode": process.returncode, "trace": leaf["trace"], "folder": str(folder)}
            compare_started = time.perf_counter()
            verification = compare_outputs(Path(rows["native"]["folder"]), Path(rows["columns"]["folder"]))
            fee = time.perf_counter()-compare_started
            for row in rows.values():
                row["phase_seconds"] += fee
                row["whole_seconds"] += fee
            pair = {"trial": trial, "rows": rows, "full_comparison_seconds_charged_each_side": fee,
                    "verification": verification}
            pairs.append(pair)
            atomic_json(output/"STATE.json", {"status": "measuring", "pairs": pairs, "counted": False})
            if rows["columns"]["whole_seconds"] > 1.1*rows["native"]["whole_seconds"]:
                raise RuntimeError("First required complete whole per-pair cost guard failed")
        medians = {label: {key: statistics.median(p["rows"][label][key] for p in pairs)
                   for key in ("phase_seconds", "whole_seconds")} for label in ("native", "columns")}
        phase_reduction = 1-medians["columns"]["phase_seconds"]/medians["native"]["phase_seconds"]
        checks = {"phase_reduction_at_least10pct": phase_reduction >= .1,
            "each_whole_at_most1_1": all(p["rows"]["columns"]["whole_seconds"] <=
                                        1.1*p["rows"]["native"]["whole_seconds"] for p in pairs),
            "whole_median_nonregressed": medians["columns"]["whole_seconds"] <= medians["native"]["whole_seconds"],
            "all_original_bindings_and_all_output_contents_exact": True}
        result = {"status": "local_export_cost_guards_passed_pending_external_and_review"
                  if all(checks.values()) else "complete_export_cost_guard_failed", "pairs": pairs,
                  "medians": medians, "checks": checks, "phase_reduction": phase_reduction,
                  "first_preparation_seconds": first_prepare,
                  "actual_complete_elapsed_seconds": time.perf_counter()-batch_started, "counted": False}
        atomic_json(output/"RESULT.json", result)
        print(json.dumps({"status": result["status"], "medians": medians, "phase_reduction": phase_reduction}))
    except BaseException as error:
        atomic_json(output/"FAILURE.json", {"error": repr(error), "pairs": pairs,
            "actual_complete_elapsed_seconds": time.perf_counter()-batch_started,
            "counted": False, "failed_outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()

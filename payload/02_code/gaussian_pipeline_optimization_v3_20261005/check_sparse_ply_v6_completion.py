"""Complete public sparse PLY checks and real in-flight signal recovery."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil
import signal
import sys
import threading

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = OUT / "v6_residual_cost_003/sparse2"
MODEL = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"
sys.path.insert(0, str(V1))
import export_admitted_map as native
from runtime import atomic_json, data_identity
from sparse_ply_oracle import sparse_ply_execution
import check_sparse_ply_oracle as generic
import check_sparse_ply_rejection as rejection


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compare_files(first, second):
    if first.stat().st_size != second.stat().st_size:
        raise AssertionError("Complete PLY sizes differ")
    checked = 0
    with first.open("rb") as left, second.open("rb") as right:
        while block := left.read(1024**2):
            if block != right.read(len(block)):
                raise AssertionError("Complete PLY contents differ")
            checked += len(block)
        if right.read(1):
            raise AssertionError("Unexpected trailing PLY data")
    return checked


def denominator(model):
    return [model.num_images(), model.num_reg_images(), model.num_points3D(),
            model.compute_num_observations()]


def invoke_fixture(module, output):
    saved = sys.argv
    try:
        sys.argv = [module.__file__, "--output", str(output)]
        module.main()
    finally:
        sys.argv = saved


def interrupt_writer(model, enabled, folder, original):
    folder.mkdir()
    path = folder / "interrupted.ply"
    active = threading.Event()
    active.set()
    evidence = {"sent": False, "file_existed_before_signal": False}

    def observe_and_signal():
        deadline = time.perf_counter() + 15
        while active.is_set() and time.perf_counter() < deadline:
            if path.exists() and active.is_set():
                evidence.update(sent=True, file_existed_before_signal=True,
                    file_bytes_at_signal=path.stat().st_size,
                    signal_during_active_writer=True)
                signal.raise_signal(signal.SIGINT)
                return
            time.sleep(.001)

    observer = threading.Thread(target=observe_and_signal, name="owned_PLY_SIGINT_observer")
    observer.start()
    captured = None
    try:
        with sparse_ply_execution(native, enabled) as trace:
            native.write_initial_ply(model, path)
    except KeyboardInterrupt as error:
        captured = type(error).__name__
    finally:
        active.clear()
        observer.join(timeout=20)
    if (observer.is_alive() or not evidence["sent"] or captured != "KeyboardInterrupt"
            or not path.is_file() or native.write_initial_ply is not original
            or not trace["original_binding_restored"] or len(trace["calls"]) != 1
            or trace["calls"][0]["status"] != "failed"
            or "KeyboardInterrupt" not in trace["calls"][0]["error"]):
        raise AssertionError("A real active complete writer did not cancel and restore")
    return {**evidence, "captured": captured, "observer_joined": True, "trace": trace,
            "preserved_file_bytes": path.stat().st_size}


def main():
    OWN.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(OWN).free < 20 * 1024**3:
        raise OSError("Registered disk headroom is unavailable")
    source_paths = [Path(__file__), Path(native.__file__), Path(generic.__file__),
        Path(rejection.__file__), Path(__file__).with_name("sparse_ply_oracle.py")]
    model_paths = sorted(MODEL.glob("*.bin"))
    if len(model_paths) != 5:
        raise AssertionError("The complete original public model membership changed")
    identities = [data_identity(path) for path in source_paths + model_paths]
    atomic_json(OWN / "CONTRACT.json", {"sources_and_inputs": identities,
        "expected_public_denominator": [360, 360, 133020, 848219],
        "denominator_fields": ["all_images", "registered_images", "points3D", "observations"],
        "same_existing_environment_path": sys.executable,
        "current_versions": {"python": sys.version, "numpy": np.__version__, "pycolmap": pycolmap.__version__},
        "old_dependency_binary_hashes_not_saved_not_invented": True,
        "component_development_not_blind_or_metric_GT": True, "new_optimization_count": 0})
    for path in source_paths:
        shutil.copyfile(path, OWN / ("source_" + path.name))
    original = native.write_initial_ply
    invoke_fixture(generic, OWN / "generic")
    invoke_fixture(rejection, OWN / "bad_ply")
    model = pycolmap.Reconstruction(MODEL)
    if denominator(model) != [360, 360, 133020, 848219]:
        raise AssertionError("Public complete model denominator differs: " + repr(denominator(model)))
    pairs = []
    for trial in range(3):
        rows = {}
        for enabled in ((False, True) if trial % 2 == 0 else (True, False)):
            name = "columns" if enabled else "native"
            folder = OWN / f"public_pair{trial}_{name}"
            folder.mkdir()
            started = time.perf_counter()
            with sparse_ply_execution(native, enabled) as trace:
                native.write_initial_ply(model, folder / "points.ply")
            if native.write_initial_ply is not original or not trace["original_binding_restored"]:
                raise AssertionError("A complete public writer did not restore its binding")
            rows[name] = {"whole_function_with_adapter_seconds": time.perf_counter() - started,
                          "trace": trace, "file": str(folder / "points.ply")}
        compared = compare_files(Path(rows["native"]["file"]), Path(rows["columns"]["file"]))
        pairs.append({"trial": trial, "rows": rows, "complete_compared_bytes": compared})
    cancellations, recoveries = {}, {}
    reference = Path(pairs[0]["rows"]["native"]["file"])
    for enabled in (False, True):
        name = "columns" if enabled else "native"
        cancellations[name] = interrupt_writer(model, enabled, OWN / ("cancel_" + name), original)
        folder = OWN / ("recovery_" + name)
        folder.mkdir()
        with sparse_ply_execution(native, enabled) as trace:
            native.write_initial_ply(model, folder / "points.ply")
        compared = compare_files(reference, folder / "points.ply")
        if not trace["original_binding_restored"] or native.write_initial_ply is not original:
            raise AssertionError("The complete new-directory recovery changed its binding")
        recoveries[name] = {"complete_writer_official_reader_and_multiset": True,
            "complete_compared_bytes": compared, "trace": trace,
            "new_directory_after_real_cancellation": True}
    checks = {
        "current_generic27_passed": read(OWN / "generic/RESULT.json")["checked"] == 27
            and all(read(OWN / "generic/RESULT.json")["checks"].values()),
        "current_all_three_actual_bad_PLY_rejections_passed": all(read(OWN / "bad_ply/RESULT.json")["checks"].values()),
        "all_three_full_public_pairs_exact": len(pairs) == 3 and all(row["complete_compared_bytes"] > 0 for row in pairs),
        "both_actual_inflight_SIGINT_and_fresh_complete_function_recoveries": len(cancellations) == len(recoveries) == 2,
        "complete_public_model_counts_preserved": denominator(model) == [360, 360, 133020, 848219],
        "all_current_sources_and_public_original_data_unchanged": [data_identity(path) for path in source_paths + model_paths] == identities,
        "all_original_bindings_restored": native.write_initial_ply is original,
    }
    output_bytes = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file())
    checks["registered_output_budget_and_disk_headroom"] = output_bytes <= 1024**3 and shutil.disk_usage(OWN).free >= 20 * 1024**3
    atomic_json(OWN / "RESULT.json", {"status": "complete_current_public_signal_and_recovery_checks_passed" if all(checks.values()) else "failed",
        "checks": checks, "public_pairs": pairs, "cancellations": cancellations, "recoveries": recoveries,
        "output_bytes_before_receipt": output_bytes, "actual_seconds_before_publication": time.perf_counter() - STARTED,
        "full_1016_export_not_rerun_old_observed_source_verified_evidence_reused": True,
        "whole_pipeline_GT_training_cross_hardware_unknown_object_quality_not_validated": True,
        "new_optimization_count": 0})
    print(json.dumps({"checks": checks, "actual_seconds_before_publication": time.perf_counter() - STARTED}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A current sparse completion protection failed")


if __name__ == "__main__":
    main()

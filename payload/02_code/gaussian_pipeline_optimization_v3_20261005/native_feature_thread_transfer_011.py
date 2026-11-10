"""Measure an existing OIIO thread control in a complete native feature stage."""

from contextlib import closing
import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

import psutil

from native_feature_gray_bank_001 import seed_database
from native_sfm_match_shards import (atomic_json, compare_databases, data_identity,
    database_summary, gpu, read, save, stop_owned)

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sft011"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
ENV_KEYS = ("OPENIMAGEIO_THREADS", "CUE_THREADS", "OPENIMAGEIO_OPTIONS")


def run_native(folder, command, limited, observer=None):
    if not isinstance(limited, bool):
        raise ValueError("The existing thread control must be an explicit boolean")
    previous = {name: os.environ.get(name) for name in ENV_KEYS}
    if any(value is not None for value in previous.values()):
        raise ValueError("The frozen native baseline requires its original unset OIIO options")
    environment = os.environ.copy()
    if limited:
        environment["OPENIMAGEIO_THREADS"] = "1"
    changes = {name: value for name, value in environment.items() if os.environ.get(name) != value}
    started = time.perf_counter()
    records, samples = [], []
    peak_threads = 0
    peak_rss = 0
    path = folder / "worker0.log"
    log = path.open("x", encoding="utf-8")
    try:
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, creationflags=flags)
        owner = psutil.Process(process.pid)
        records.append({"process": process, "pid": process.pid, "creation_time": owner.create_time(),
                        "command": command, "environment_changes_only": changes})
        next_gpu = started
        while process.poll() is None:
            now = time.perf_counter()
            if now - started > 120:
                raise RuntimeError("The registered native feature time bound was exceeded")
            try:
                peak_threads = max(peak_threads, owner.num_threads())
                peak_rss = max(peak_rss, owner.memory_info().rss)
            except psutil.NoSuchProcess:
                if process.poll() is None:
                    raise RuntimeError("An owned native feature process vanished without an exit status")
                break
            if observer is not None:
                observer(process, path)
            if now >= next_gpu:
                samples.append({"seconds": now - started, **gpu()})
                next_gpu = now + 2
            time.sleep(.1)
        if process.returncode != 0:
            raise ChildProcessError("The actual native feature process failed with exit " + str(process.returncode))
    finally:
        stop_owned(records)
        log.close()
        save(folder / "NATIVE_PROCESSES.json", [{key: value for key, value in row.items() if key != "process"} | {"returncode": row["process"].returncode} for row in records])
        save(folder / "GPU_SAMPLES.json", samples)
        if previous != {name: os.environ.get(name) for name in ENV_KEYS}:
            raise AssertionError("The private child thread control modified the original parent environment")
    return {"complete_native_process_seconds": time.perf_counter() - started,
        "all_native_processes_waited": True, "limited": limited, "environment_changes_only": changes,
        "original_parent_environment": previous, "peak_all_native_threads_not_OIIO_thread_count": peak_threads,
        "peak_native_RSS_bytes": peak_rss}


def entry(folder, limited, template, names):
    folder.mkdir()
    started = time.perf_counter()
    database = folder / "database.db"
    seed_database(database)
    command = template.copy()
    command[command.index("--database_path") + 1] = str(database)
    native = run_native(folder, command, limited)
    with closing(sqlite3.connect(database)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise AssertionError("The private complete feature database failed integrity checking")
        if connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
            raise RuntimeError("The owned native feature writer did not fully drain")
    summary = database_summary(database, names)
    atomic_json(folder / "FEATURES.json", summary)
    gc.collect()
    result = {"limited": limited, "database": str(database), "summary": summary, "native": native,
        "complete_metadata_to_ready_features_seconds": time.perf_counter() - started,
        "original1002_JPEG_and_single_native_GPU_no_bank_or_shard": True,
        "includes_metadata_copy_delete_startup_native_all_images_wait_summary_publish_and_drain": True}
    save(folder / "ENTRY.json", result)
    print(json.dumps({"limited": limited, "complete_seconds": result["complete_metadata_to_ready_features_seconds"],
        "peak_all_native_threads_not_OIIO": native["peak_all_native_threads_not_OIIO_thread_count"]}), flush=True)
    return result


def main():
    started = time.perf_counter()
    output = Path(sys.argv[1]).resolve()
    output.relative_to(OWN)
    order = sys.argv[2] if len(sys.argv) > 2 else "AB"
    if order not in ("AB", "BA") or output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise ValueError("A fresh registered private full native feature transfer pair is required")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    template = next(row for row in read(OLD / "STATE.json")["stages"] if row["name"] == "features")["command"].copy()
    template[0] = str((ROOT / template[0]).resolve())
    for option, value in (("--FeatureExtraction.gpu_index", "0"), ("--FeatureExtraction.num_threads", "8"),
                          ("--FeatureExtraction.max_image_size", "1600"), ("--SiftExtraction.max_num_features", "8192")):
        if template[template.index(option) + 1] != value:
            raise AssertionError("The frozen original complete native feature parameters changed")
    names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    images = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup010/p2/PROTOCOL.json")["full_images"]
    if len(names) != 1002 or [Path(row["path"]).name for row in images] != names:
        raise AssertionError("The full frozen native feature population changed")
    paths = [Path(__file__), CODE / "native_feature_gray_bank_001.py", CODE / "native_sfm_match_shards.py",
        CODE / "dataset_sampling_carrier.py", V1 / "run_sfm_preflight.py", V1 / "runtime.py", Path(template[0]),
        Path(template[0]).parent / "OpenImageIO.dll", Path(template[0]).parent / "OpenImageIO_Util.dll"]
    sources = [data_identity(path) for path in paths]
    inputs = [data_identity(OLD / name) for name in ("features.db", "STATE.json", "CONTRACT.json", "image_list.txt")]
    save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs, "images": images,
        "template": template, "order": order, "new_optimization_count": 0,
        "existing_R123_thread_control_transferred_not_new_mechanism": True,
        "task": str(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/EXISTING_OIIO_FEATURE_TRANSFER_011_TASK.md")})
    for path in paths:
        if path.suffix == ".py":
            shutil.copyfile(path, output / "sources" / path.name)
    entries = {}
    try:
        gpu_before = gpu()
        for side in order:
            entries[side] = entry(output / side, side == "B", template, names)
        baseline, candidate = entries["A"], entries["B"]
        current_oracle = compare_databases(baseline["database"], candidate["database"])
        original_baseline = compare_databases(OLD / "features.db", baseline["database"])
        original_candidate = compare_databases(OLD / "features.db", candidate["database"])
        checks = {
            "all12_current_native_columns_and_BLOBs_exact": current_oracle["complete_values_exact"],
            "all_original_baseline_candidate_native_columns_exact": original_baseline["complete_values_exact"] and original_candidate["complete_values_exact"],
            "whole1002_names_and1779059_features_and_summary_exact": baseline["summary"] == candidate["summary"] and len(candidate["summary"]["keypoint_counts"]) == 1002 and sum(candidate["summary"]["keypoint_counts"].values()) == 1779059,
            "only_child_candidate_thread_environment_changed": baseline["native"]["environment_changes_only"] == {} and candidate["native"]["environment_changes_only"] == {"OPENIMAGEIO_THREADS": "1"},
            "parent_environment_original_unset_and_processes_drained": not any(os.environ.get(name) is not None for name in ENV_KEYS) and baseline["native"]["all_native_processes_waited"] and candidate["native"]["all_native_processes_waited"],
            "all_frozen_sources_DLLs_inputs_and1002_original_images_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs + images),
            "registered_disk_and_output_budget_preserved": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 4 * 1024 ** 3,
        }
        result = {"status": "complete_existing_thread_control_transfer_pair_not_accepted",
            "checks": checks, "baseline": baseline, "candidate": candidate, "sources": sources, "inputs": inputs,
            "current_native_oracle": current_oracle, "original_baseline_oracle": original_baseline,
            "original_candidate_oracle": original_candidate, "gpu_before": gpu_before, "gpu_after": gpu(),
            "raw_whole_ratio": candidate["complete_metadata_to_ready_features_seconds"] / baseline["complete_metadata_to_ready_features_seconds"],
            "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0}
        save(output / "RESULT.json", result)
        print(json.dumps({"checks": checks, "ratio": result["raw_whole_ratio"], "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("An original full native feature transfer check failed")
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "completed_sides": entries, "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()

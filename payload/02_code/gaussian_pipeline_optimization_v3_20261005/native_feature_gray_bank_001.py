"""Prepare exact native grayscale inputs for an unchanged GPU SIFT consumer."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback

import numpy as np
import psutil
import pycolmap

from native_sfm_match_shards import (atomic_json, compare_databases, data_identity,
    database_summary, gpu, read, readonly, run_native_jobs, save)

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sfg001"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
DIAG = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup010"


def prepare_one(identity, target):
    source = Path(identity["path"])
    if data_identity(source) != identity:
        raise AssertionError("A bank input differs from its frozen image identity")
    target = Path(target)
    bitmap = pycolmap.Bitmap.read(source, False)
    if bitmap is None or bitmap.is_empty or not bitmap.is_grey:
        raise ValueError("A native bank input cannot be read as grayscale")
    array = bitmap.to_array()
    if array.dtype != np.uint8 or array.ndim != 2 or not array.flags.c_contiguous:
        raise ValueError("The native grayscale array has an unsupported representation")
    height, width = array.shape
    if [width, height] != [bitmap.width, bitmap.height]:
        raise AssertionError("The native bitmap and array dimensions disagree")
    header = ("P5\n" + str(width) + " " + str(height) + "\n255\n").encode("ascii")
    with target.open("xb") as stream:
        if stream.write(header) != len(header) or stream.write(memoryview(array).cast("B")) != array.nbytes:
            raise OSError("A native PGM bank payload was not fully written")
    current = psutil.Process()
    return {"name": source.name, "dimensions": [width, height], "payload_bytes": array.nbytes,
        "file_bytes": target.stat().st_size, "pid": os.getpid(), "creation_time": current.create_time()}


def prepare_bank(folder, identities):
    started = time.perf_counter()
    folder.mkdir()
    names = [Path(row["path"]).name for row in identities]
    if len(set(names)) != len(names) or not names or any(Path(name).name != name for name in names):
        raise ValueError("The flat bank image names must be unique and valid")
    rows = [None] * len(identities)
    pending = {}
    next_index = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        try:
            while next_index < len(identities) or pending:
                while next_index < len(identities) and len(pending) < 16:
                    future = executor.submit(prepare_one, identities[next_index], str(folder / names[next_index]))
                    pending[future] = next_index
                    next_index += 1
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    rows[pending.pop(future)] = future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    result = {"rows": rows, "workers": 8, "maximum_inflight": 16,
        "complete_cold_bank_seconds_including_pool_startup_and_shutdown": time.perf_counter() - started,
        "all_workers_waited_and_no_reused_bank": True}
    save(folder.parent / "BANK.json", result)
    return result


def bank_pixels(folder, identities):
    started = time.perf_counter()
    compared_bytes = 0
    for row in identities:
        original = pycolmap.Bitmap.read(Path(row["path"]), False)
        bank = pycolmap.Bitmap.read(folder / Path(row["path"]).name, False)
        if original is None or bank is None:
            raise ValueError("The existing native reader cannot identify the bank image format")
        a, b = original.to_array(), bank.to_array()
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise AssertionError("A complete bank grayscale payload differs from the original native pixels")
        compared_bytes += a.nbytes
    return {"whole_native_gray_pixels_exact": True, "images": len(identities),
        "directly_compared_pixel_bytes": compared_bytes, "seconds": time.perf_counter() - started,
        "not_a_file_hash_or_sampled_pixel_oracle": True}


def seed_database(path):
    shutil.copyfile(OLD / "features.db", path)
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("DELETE FROM keypoints")
            connection.execute("DELETE FROM descriptors")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise AssertionError("The private metadata database failed native integrity checking")
    compare_databases(OLD / "features.db", path, exclude=("keypoints", "descriptors"))


def verify_feature_subset(path, names):
    compared_bytes = 0
    with closing(readonly(OLD / "features.db")) as original, closing(readonly(path)) as current:
        image_ids = dict(original.execute("SELECT name,image_id FROM images").fetchall())
        for table in ("keypoints", "descriptors"):
            if current.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] != len(names):
                raise AssertionError("A native feature component has missing or extra image rows")
            for name in names:
                expected = original.execute("SELECT * FROM " + table + " WHERE image_id=?", (image_ids[name],)).fetchone()
                actual = current.execute("SELECT * FROM " + table + " WHERE image_id=?", (image_ids[name],)).fetchone()
                if expected != actual:
                    raise AssertionError("The native feature component differs from original columns for " + name + " in " + table)
                compared_bytes += sum(len(value) for value in actual if isinstance(value, bytes))
    return {"original_complete_subset_feature_columns_exact": True, "images": len(names),
        "directly_compared_BLOB_bytes": compared_bytes}


def entry(folder, candidate, identities, template, full_names, component=False):
    folder.mkdir()
    started = time.perf_counter()
    database = folder / "database.db"
    seed_database(database)
    names = [Path(row["path"]).name for row in identities]
    bank = prepare_bank(folder / "images", identities) if candidate else None
    image_path = folder / "images" if candidate else Path(identities[0]["path"]).parent
    image_list = folder / "image_list.txt"
    image_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    command = template.copy()
    for option, value in (("--database_path", database), ("--image_path", image_path), ("--image_list_path", image_list)):
        command[command.index(option) + 1] = str(value)
    native = run_native_jobs(folder, [command])
    with closing(sqlite3.connect(database)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise AssertionError("The complete native feature output failed integrity checking")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint[0] != 0:
            raise RuntimeError("The owned native database writer did not fully drain")
    if component:
        summary = {"component_images": len(names), "not_ready_full_matching_database": True}
    else:
        summary = database_summary(database, full_names)
    atomic_json(folder / "FEATURES.json", summary)
    gc.collect()
    result = {"candidate": candidate, "database": str(database), "summary": summary,
        "complete_metadata_to_ready_features_seconds": time.perf_counter() - started,
        "bank": bank, "native": native, "component_only": component,
        "includes_seed_copy_delete_bank_startup_decode_write_drain_native_summary_publish": True}
    save(folder / "ENTRY.json", result)
    print(json.dumps({"candidate": candidate, "images": len(names),
        "complete_seconds": result["complete_metadata_to_ready_features_seconds"],
        "bank_seconds": bank["complete_cold_bank_seconds_including_pool_startup_and_shutdown"] if bank else None}), flush=True)
    return result


def main():
    started = time.perf_counter()
    output = Path(sys.argv[1]).resolve()
    output.relative_to(OWN)
    mode = sys.argv[2]
    order = sys.argv[3] if len(sys.argv) > 3 else "AB"
    if mode not in ("component", "full") or order not in ("AB", "BA") or output.exists():
        raise ValueError("A fresh registered bank diagnostic or complete pair is required")
    if shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise RuntimeError("The registered disk reserve is not available")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    template = next(row for row in read(OLD / "STATE.json")["stages"] if row["name"] == "features")["command"].copy()
    template[0] = str((ROOT / template[0]).resolve())
    if template[1] != "feature_extractor" or template[template.index("--FeatureExtraction.gpu_index") + 1] != "0" or template[template.index("--FeatureExtraction.num_threads") + 1] != "8":
        raise AssertionError("The original single native GPU feature interface changed")
    identities = read(DIAG / "p2/PROTOCOL.json")["full_images"]
    full_names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    if len(identities) != 1002 or [Path(row["path"]).name for row in identities] != full_names or pycolmap.__version__ != "4.1.1":
        raise AssertionError("The full frozen image denominator or existing module changed")
    if mode == "component":
        identities = [identities[round(index * 1001 / 31)] for index in range(32)]
    paths = [Path(__file__), CODE / "native_sfm_match_shards.py", V1 / "run_sfm_preflight.py", V1 / "runtime.py", Path(pycolmap._core.__file__), Path(template[0])]
    sources = [data_identity(path) for path in paths]
    inputs = [data_identity(OLD / name) for name in ("features.db", "STATE.json", "CONTRACT.json", "image_list.txt")]
    save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs, "images": identities,
        "template": template, "mode": mode, "order": order, "workers": 8, "inflight": 16,
        "full_current_denominator": 1002, "new_optimization_count": 0,
        "task": str(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R166_EXACT_GRAY_BANK_TASK.md")})
    for path in paths:
        if path.suffix == ".py":
            shutil.copyfile(path, output / "sources" / path.name)
    entries = {}
    try:
        gpu_before = gpu()
        for side in order:
            entries[side] = entry(output / side, side == "B", identities, template, full_names, mode == "component")
        baseline, candidate = entries["A"], entries["B"]
        pixel_oracle = bank_pixels(output / "B/images", identities)
        native_oracle = compare_databases(baseline["database"], candidate["database"])
        if mode == "full":
            original_oracle = compare_databases(OLD / "features.db", baseline["database"])
            original_candidate_oracle = compare_databases(OLD / "features.db", candidate["database"])
        else:
            original_oracle = compare_databases(OLD / "features.db", baseline["database"], exclude=("keypoints", "descriptors"))
            original_candidate_oracle = verify_feature_subset(candidate["database"], [Path(row["path"]).name for row in identities])
            verify_feature_subset(baseline["database"], [Path(row["path"]).name for row in identities])
        checks = {
            "whole_native_bank_pixels_exact": pixel_oracle["whole_native_gray_pixels_exact"],
            "all_native_baseline_candidate_columns_exact": native_oracle["complete_values_exact"],
            "original_metadata_and_all_active_feature_columns_exact": True,
            "all_original_sources_and_inputs_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs + identities),
            "single_GPU_original_params_all_job_processes_waited": baseline["native"]["native_processes"] == 1 and candidate["native"]["native_processes"] == 1 and baseline["native"]["all_native_processes_actually_waited"] and candidate["native"]["all_native_processes_actually_waited"],
            "cold_bank_original_population_and_workers_drained": len(candidate["bank"]["rows"]) == len(identities) and candidate["bank"]["all_workers_waited_and_no_reused_bank"],
            "registered_disk_and_full_candidate_output_budget_preserved": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 24 * 1024 ** 3,
        }
        result = {"status": "native_bank_component0" if mode == "component" else "complete_native_feature_pair_not_accepted",
            "checks": checks, "baseline": baseline, "candidate": candidate, "pixel_oracle": pixel_oracle,
            "native_oracle": native_oracle, "original_baseline_oracle": original_oracle,
            "original_candidate_oracle": original_candidate_oracle,
            "sources": sources, "inputs": inputs, "gpu_before": gpu_before, "gpu_after": gpu(),
            "raw_whole_ratio": candidate["complete_metadata_to_ready_features_seconds"] / baseline["complete_metadata_to_ready_features_seconds"],
            "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0}
        save(output / "RESULT.json", result)
        print(json.dumps({"checks": checks, "ratio": result["raw_whole_ratio"], "component_only": mode == "component", "count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A preregistered exact native bank check failed")
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "completed_sides": entries, "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()

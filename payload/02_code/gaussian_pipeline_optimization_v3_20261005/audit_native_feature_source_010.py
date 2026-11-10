"""Audit full feature logs and existing native image preparation without SIFT."""

from contextlib import closing
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request

import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup010"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
sys.path.insert(0, str(V1))
from runtime import data_identity
from native_sfm_match_shards import gpu, read, save


def official_source(folder, relative):
    url = "https://raw.githubusercontent.com/colmap/colmap/a0d785f/" + relative
    started = time.perf_counter()
    path = folder / Path(relative).name
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Codex-bounded-official-source-audit"})
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError("An official fixed-version source did not return HTTP 200")
            raw = response.read(2 * 1024 ** 2 + 1)
            if len(raw) > 2 * 1024 ** 2:
                raise RuntimeError("The official source exceeds the registered bounded read")
        text = raw.decode("utf-8")
        if "Copyright (c), ETH Zurich and UNC Chapel Hill." not in text or "Redistribution and use" not in text:
            raise ValueError("The official source copyright and redistribution notice are missing")
        with path.open("xb") as stream:
            stream.write(raw)
        return {"url": url, "status": "ok", "source": data_identity(path), "lines": len(text.splitlines()),
                "original_copyright_and_license_preserved": True, "seconds": time.perf_counter() - started}
    except (urllib.error.URLError, UnicodeError, RuntimeError, ValueError) as error:
        return {"url": url, "status": "failed", "error": repr(error), "seconds": time.perf_counter() - started}


def original_log_rows(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    records = []
    for index, line in enumerate(lines):
        match = re.search(r"^I(\d{8}) (\d{2}:\d{2}:\d{2}\.\d{6}).*Processed file \[(\d+)/(\d+)\]", line)
        if not match:
            continue
        block = lines[index + 1:index + 8]
        name = next(value.split("Name:", 1)[1].strip() for value in block if "Name:" in value)
        keys = int(next(value.split("Features:", 1)[1].strip().split()[0] for value in block if "Features:" in value))
        timestamp = datetime.strptime(match[1] + " " + match[2], "%Y%m%d %H:%M:%S.%f").timestamp()
        records.append({"writer_index": int(match[3]), "denominator": int(match[4]), "name": name,
                        "keypoints": keys, "timestamp_seconds_local_log_not_UTC": timestamp, "source_line": index + 1})
    return records, lines


def main():
    started = time.perf_counter()
    output = OWN / "p0"
    if output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise ValueError("A fresh private native image source diagnostic and disk reserve are required")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    sources = [data_identity(Path(__file__)), data_identity(V1 / "run_sfm_preflight.py"),
               data_identity(Path(pycolmap._core.__file__))]
    paths = [OLD / name for name in ("features.db", "features.log", "FEATURES.json", "STATE.json", "CONTRACT.json", "image_list.txt")]
    inputs = [data_identity(path) for path in paths]
    contract = read(OLD / "CONTRACT.json")
    images = [dict(row, path=str((ROOT / row["path"]).resolve())) for row in contract["input_images"]]
    names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    if len(images) != 1002 or len(names) != 1002 or len(set(names)) != 1002 or pycolmap.__version__ != "4.1.1":
        raise AssertionError("The complete original image or existing module denominator changed")
    template = next(row for row in read(OLD / "STATE.json")["stages"] if row["name"] == "features")
    save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs, "images": images,
        "exact_existing_native_feature_command_not_executed": template["command"], "guard_seconds": 180,
        "full_images": 1002, "component_only_existing_CPU_pycolmap_Bitmap_not_CLI_profile": True,
        "task": str(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/NATIVE_FEATURE_SOURCE_COST_010_TASK.md"),
        "new_optimization_count": 0, "observed_development_inputs_not_blind_GT": True})
    shutil.copyfile(Path(__file__), output / "sources" / Path(__file__).name)
    rows = []
    try:
        downloads = [official_source(output / "sources", relative) for relative in (
            "src/colmap/controllers/feature_extraction.cc", "src/colmap/controllers/image_reader.cc",
            "src/colmap/scene/database_sqlite.cc", "src/colmap/feature/sift.cc", "src/colmap/sensor/bitmap.cc")]
        save(output / "OFFICIAL_SOURCE_FETCH.json", {"downloads": downloads,
            "prior_web_failures": ["src/colmap/util/bitmap.cc HTTP404", "src/colmap/image/bitmap.cc HTTP404"],
            "correct_bitmap_path_from_fixed_ImageReader_header": "colmap/sensor/bitmap.h",
            "no_master_or_unpinned_source_substitution": True})
        if not all(value["status"] == "ok" for value in downloads):
            raise RuntimeError("Fixed official sources could not all be retained; inspect failed URL records")
        controller = (output / "sources/feature_extraction.cc").read_text(encoding="utf-8")
        bitmap_source = (output / "sources/bitmap.cc").read_text(encoding="utf-8")
        if "bitmap->Thumbnail(max_image_size_)" not in controller or "std::round(width_ * scale)" not in bitmap_source:
            raise AssertionError("The original native thumbnail interface or rounding contract changed")
        log_rows, log_lines = original_log_rows(OLD / "features.log")
        with closing(sqlite3.connect("file:" + (OLD / "features.db").as_posix() + "?mode=ro", uri=True)) as connection:
            keys = dict(connection.execute("SELECT images.name,keypoints.rows FROM images JOIN keypoints USING(image_id)").fetchall())
            schema = dict(connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall())
        if len(log_rows) != 1002 or {row["name"]: row["keypoints"] for row in log_rows} != keys:
            raise AssertionError("The entire original writer log differs from native keypoint populations")
        before_gpu = gpu()
        image_source_seconds = 0.
        for index, identity in enumerate(images):
            path = Path(identity["path"])
            verify_started = time.perf_counter()
            if data_identity(path) != identity:
                raise AssertionError("An original current JPEG differs from its frozen feature contract")
            image_source_seconds += time.perf_counter() - verify_started
            begin = time.perf_counter()
            bitmap = pycolmap.Bitmap.read(path, False)
            decoded = time.perf_counter()
            if bitmap is None or bitmap.is_empty or not bitmap.is_grey:
                raise ValueError("An original complete native image could not be read as grayscale")
            dimensions = [bitmap.width, bitmap.height]
            orientation = bitmap.exif_orientation()
            resize_started = time.perf_counter()
            scale = bitmap.thumbnail(1600)
            resized = time.perf_counter()
            expected_scale = min(1., 1600 / max(dimensions))
            expected = [int(value * expected_scale + .5) for value in dimensions]
            if [bitmap.width, bitmap.height] != expected or scale != expected_scale:
                raise AssertionError("The native thumbnail dimensions or scale differ from the original round contract")
            rows.append({"index": index, "name": path.name, "dimensions": dimensions,
                "thumbnail_dimensions": [bitmap.width, bitmap.height], "orientation": orientation,
                "native_read_seconds": decoded - begin, "native_thumbnail_seconds": resized - resize_started,
                "scaled_bitmap_bytes": bitmap.width * bitmap.height * bitmap.channels})
            del bitmap
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed_original_images": index + 1, "component_seconds": sum(row["native_read_seconds"] + row["native_thumbnail_seconds"] for row in rows)}), flush=True)
        gc.collect()
        save(output / "NATIVE_PREPARATION_ROWS.json", rows)
        save(output / "ORIGINAL_COMPLETE_WRITER_ROWS.json", log_rows)
        deltas = [log_rows[index]["timestamp_seconds_local_log_not_UTC"] - log_rows[index - 1]["timestamp_seconds_local_log_not_UTC"] for index in range(1, len(log_rows))]
        read_seconds = sum(row["native_read_seconds"] for row in rows)
        resize_seconds = sum(row["native_thumbnail_seconds"] for row in rows)
        checks = {
            "all_original1002_writer_indices_names_keypoints_and_database_rows_match": [row["writer_index"] for row in log_rows] == list(range(1, 1003)) and all(row["denominator"] == 1002 for row in log_rows) and sorted(keys) == sorted(names) and sum(keys.values()) == 1779059,
            "original_GPU_extractor_really_present_in_entire_raw_log": any("Creating SIFT GPU feature extractor" in line for line in log_lines),
            "five_fixed_official_sources_with_full_copyright_and_license_saved": len(downloads) == 5 and all(row["status"] == "ok" and row["original_copyright_and_license_preserved"] for row in downloads),
            "whole1002_existing_native_Bitmap_read_and_original_thumbnail_dimensions": len(rows) == 1002 and [row["name"] for row in rows] == names and all(row["thumbnail_dimensions"] == [1600, 1222] for row in rows),
            "all1002_current_JPEG_bytes_still_exact_frozen_contract": all(data_identity(row["path"]) == row for row in images),
            "all_current_module_entry_original_database_and_metadata_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs),
            "original_native_schema_preserved_not_written_by_component": len(schema) == 12,
            "registered_disk_and_whole_GPU_reserve_preserved": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 2 * 1024 ** 3,
        }
        result = {"status": "full_original_feature_source_and_native_preparation_component_diagnostic0", "checks": checks,
            "sources": sources, "inputs": inputs, "official_sources": downloads,
            "source_schema": schema, "original_feature_stage_historical_seconds_not_new_benchmark": template["elapsed_seconds"],
            "old_writer_intervals_not_individual_native_stage_latency": {"count": len(deltas), "median_seconds": statistics.median(deltas), "total_writer_timestamp_span_seconds": log_rows[-1]["timestamp_seconds_local_log_not_UTC"] - log_rows[0]["timestamp_seconds_local_log_not_UTC"], "largest_seconds": max(deltas)},
            "actual_existing_pycolmap_native_component_costs": {"read_seconds": read_seconds, "thumbnail_seconds": resize_seconds,
                "sequential_sum_seconds": read_seconds + resize_seconds, "JPEG_identity_check_seconds_not_component_cost": image_source_seconds,
                "module_version": pycolmap.__version__, "has_cuda": pycolmap.has_cuda, "not_actual_CLI_GPU_or_writer_profile_or_savings": True},
            "original_orientation_values": sorted(set(str(row["orientation"]) for row in rows)),
            "gpu_before": before_gpu, "gpu_after": gpu(), "elapsed_seconds": time.perf_counter() - started,
            "SIFT_matching_mapping_training_new_algorithm_or_quality_effect_not_run": True, "root_count": 36, "new_optimization_count": 0}
        save(output / "RESULT.json", result)
        print(json.dumps({"checks": checks, "component": result["actual_existing_pycolmap_native_component_costs"], "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The full original image source diagnostic failed")
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "completed_image_count": len(rows), "sources": sources, "inputs": inputs, "new_optimization_count": 0})
        if rows and not (output / "NATIVE_PREPARATION_ROWS.json").exists():
            save(output / "NATIVE_PREPARATION_ROWS.json", rows)
        raise


if __name__ == "__main__":
    main()

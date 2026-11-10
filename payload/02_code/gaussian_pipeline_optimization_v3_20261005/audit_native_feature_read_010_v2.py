"""Separate full native image reading from a fixed thumbnail diagnostic."""

from contextlib import closing
import gc
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import urllib.request

import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup010"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
sys.path.insert(0, str(V1))
from runtime import data_identity
from audit_native_feature_source_010 import original_log_rows
from native_sfm_match_shards import gpu, read, readonly, save


def official_binding_with_license(folder):
    started = time.perf_counter()
    records = []
    for relative in ("COPYING.txt", "src/pycolmap/sensor/bitmap.cc"):
        url = "https://raw.githubusercontent.com/colmap/colmap/a0d785f/" + relative
        request = urllib.request.Request(url, headers={"User-Agent": "Codex-bounded-official-source-audit"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(2 * 1024 ** 2 + 1)
            if response.status != 200 or len(raw) > 2 * 1024 ** 2:
                raise RuntimeError("A fixed official binding or license fetch exceeded its contract")
        text = raw.decode("utf-8")
        if relative == "COPYING.txt" and ("Copyright (c), ETH Zurich and UNC Chapel Hill." not in text or "Redistribution and use" not in text):
            raise ValueError("The fixed official companion license is not the expected original notice")
        path = folder / Path(relative).name
        with path.open("xb") as stream:
            stream.write(raw)
        records.append({"url": url, "source": data_identity(path), "lines": len(text.splitlines())})
    return {"status": "ok", "source": records[1]["source"], "companion_license": records[0],
        "url": records[1]["url"], "original_copyright_and_license_preserved": True,
        "original_binding_has_no_per_file_notice": True, "seconds": time.perf_counter() - started}


def main():
    started = time.perf_counter()
    output = OWN / "p2"
    if output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise ValueError("A fresh private diagnostic with the registered disk reserve is required")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    previous = read(OWN / "p0/PROTOCOL.json")
    images = previous["images"]
    names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    fetches = read(OWN / "p0/OFFICIAL_SOURCE_FETCH.json")["downloads"]
    sources = [data_identity(Path(__file__)), *previous["sources"],
               *[row["source"] for row in fetches]]
    inputs = previous["inputs"]
    indices = [round(index * 1001 / 31) for index in range(32)]
    if len(images) != 1002 or len(set(indices)) != 32 or pycolmap.__version__ != "4.1.1":
        raise AssertionError("The full denominator, fixed sample, or existing version changed")
    if not all(data_identity(row["path"]) == row for row in sources + inputs):
        raise AssertionError("A retained diagnostic source or frozen input changed")
    save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs,
        "full_images": images, "fixed_thumbnail_indices": indices, "maximum_seconds": 180,
        "task": str(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/NATIVE_FEATURE_SOURCE_COST_010_V2_TASK.md"),
        "native_CPU_interface_not_CLI_cost_profile": True, "new_optimization_count": 0})
    shutil.copyfile(Path(__file__), output / "sources" / Path(__file__).name)
    rows, thumbnail_rows = [], []
    try:
        binding = official_binding_with_license(output / "sources")
        save(output / "BINDING_SOURCE_FETCH.json", binding)
        if binding["status"] != "ok":
            raise RuntimeError("The fixed official Python binding source could not be retained")
        sources.append(binding["source"])
        sources.append(binding["companion_license"]["source"])
        text = Path(binding["source"]["path"]).read_text(encoding="utf-8")
        if '"read"' not in text or '"thumbnail"' not in text:
            raise AssertionError("The official Bitmap binding interfaces changed")
        log_rows, log_lines = original_log_rows(OLD / "features.log")
        with closing(readonly(OLD / "features.db")) as connection:
            keys = dict(connection.execute("SELECT images.name,keypoints.rows FROM images JOIN keypoints USING(image_id)").fetchall())
            schema = dict(connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall())
        if len(log_rows) != 1002 or {row["name"]: row["keypoints"] for row in log_rows} != keys:
            raise AssertionError("The original whole log and actual feature populations differ")
        save(output / "ORIGINAL_COMPLETE_WRITER_ROWS.json", log_rows)
        before_gpu = gpu()
        identity_seconds = 0.
        for index, identity in enumerate(images):
            verify_started = time.perf_counter()
            if data_identity(identity["path"]) != identity:
                raise AssertionError("An original JPEG differs from its frozen complete feature contract")
            identity_seconds += time.perf_counter() - verify_started
            begin = time.perf_counter()
            bitmap = pycolmap.Bitmap.read(Path(identity["path"]), False)
            decoded = time.perf_counter()
            if bitmap is None or bitmap.is_empty or not bitmap.is_grey:
                raise ValueError("An original native grayscale read failed")
            rows.append({"index": index, "name": Path(identity["path"]).name,
                "dimensions": [bitmap.width, bitmap.height], "orientation": bitmap.exif_orientation(),
                "native_read_seconds": decoded - begin, "channels": bitmap.channels})
            del bitmap
            if (index + 1) % 100 == 0:
                save(output / ("READ_CHECKPOINT_" + str(index + 1) + ".json"), rows)
                print(json.dumps({"completed_full_original_reads": index + 1,
                    "native_read_seconds": sum(row["native_read_seconds"] for row in rows)}), flush=True)
        save(output / "FULL_NATIVE_READ_ROWS.json", rows)
        for index in indices:
            bitmap = pycolmap.Bitmap.read(Path(images[index]["path"]), False)
            if bitmap is None:
                raise ValueError("A fixed thumbnail sample could not be read")
            dimensions = [bitmap.width, bitmap.height]
            begin = time.perf_counter()
            scale = bitmap.thumbnail(1600)
            elapsed = time.perf_counter() - begin
            expected_scale = min(1., 1600 / max(dimensions))
            expected = [int(value * expected_scale + .5) for value in dimensions]
            if [bitmap.width, bitmap.height] != expected or scale != expected_scale:
                raise AssertionError("The fixed sample differs from the native rounding contract")
            thumbnail_rows.append({"index": index, "name": rows[index]["name"],
                "original_dimensions": dimensions, "thumbnail_dimensions": expected,
                "native_thumbnail_seconds": elapsed, "scale": scale})
            del bitmap
        save(output / "FIXED32_THUMBNAIL_ROWS.json", thumbnail_rows)
        gc.collect()
        checks = {
            "whole1002_writer_indices_names_keypoints_and_database_rows_match": [row["writer_index"] for row in log_rows] == list(range(1, 1003)) and all(row["denominator"] == 1002 for row in log_rows) and sorted(keys) == sorted(names) and sum(keys.values()) == 1779059,
            "whole1002_native_read_names_dimensions_gray_complete": len(rows) == 1002 and [row["name"] for row in rows] == names and all(row["dimensions"] == [1624, 1240] and row["channels"] == 1 for row in rows),
            "only_fixed32_original_thumbnail_dimensions_verified": len(thumbnail_rows) == 32 and [row["index"] for row in thumbnail_rows] == indices and all(row["thumbnail_dimensions"] == [1600, 1222] for row in thumbnail_rows),
            "all1002_JPEG_bytes_still_exact_frozen_contract": all(data_identity(row["path"]) == row for row in images),
            "all_current_and_retained_sources_and_inputs_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs),
            "official_binding_original_license_saved_and_no_GIL_release_declared": binding["original_copyright_and_license_preserved"] and "gil_scoped_release" not in text and "call_guard" not in text,
            "original_GPU_extractor_present_and_schema12_preserved": any("Creating SIFT GPU feature extractor" in line for line in log_lines) and len(schema) == 12,
            "registered_disk_and_output_budget_preserved": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 2 * 1024 ** 3,
        }
        result = {"status": "whole_native_read_and_fixed32_thumbnail_source_diagnostic0", "checks": checks,
            "sources": sources, "inputs": inputs, "binding_source": binding,
            "read1002_seconds": sum(row["native_read_seconds"] for row in rows),
            "thumbnail32_seconds": sum(row["native_thumbnail_seconds"] for row in thumbnail_rows),
            "thumbnail32_median_seconds": statistics.median(row["native_thumbnail_seconds"] for row in thumbnail_rows),
            "JPEG_identity_seconds_outside_component": identity_seconds,
            "module_version": pycolmap.__version__, "module_has_cuda": pycolmap.has_cuda,
            "original_orientation_values": sorted(set(str(row["orientation"]) for row in rows)),
            "gpu_before": before_gpu, "gpu_after": gpu(), "elapsed_seconds": time.perf_counter() - started,
            "p0_timed_out_and_full1002_thumbnail_not_completed": True,
            "not_CLI_profile_savings_full_SIFT_or_generalization_verification": True,
            "root_count": 36, "new_optimization_count": 0}
        save(output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("checks", "read1002_seconds", "thumbnail32_seconds", "original_orientation_values", "new_optimization_count")}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A preregistered separate read and thumbnail diagnostic check failed")
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "completed_read_count": len(rows), "completed_thumbnail_count": len(thumbnail_rows), "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()

"""Preserve pinned official matching sources for a read-only locality audit."""

import json
import shutil
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup013"
PREFIX = "https://raw.githubusercontent.com/colmap/colmap/a0d785f/src/colmap/"
FILES = (
    "controllers/feature_matching.h",
    "controllers/feature_matching.cc",
    "controllers/feature_matching_utils.h",
    "controllers/feature_matching_utils.cc",
    "controllers/pairing.h",
    "controllers/pairing.cc",
    "util/cache.h",
    "exe/feature.cc",
    "estimators/two_view_geometry.cc",
)


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(BASE)
    if output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024**3:
        raise ValueError("A fresh private source-audit directory is required")
    output.mkdir(parents=True)
    started = time.perf_counter()
    original = data_identity(Path(__file__))
    save(output / "PROTOCOL.json", {
        "source": original, "fixed_official_commit": "a0d785f",
        "files": FILES, "prior_web_failures": [
            "src/colmap/feature/pairing.h HTTP404",
            "src/colmap/feature/pairing.cc HTTP404",
        ], "correct_pairing_path_from_original_feature_matching_header": "controllers/pairing.h",
        "parent_count": 36, "new_optimization_count": 0,
        "diagnostic_only_no_matching_mapper_BA_training_or_environment_change": True,
    })
    sources = output / "sources"
    sources.mkdir()
    shutil.copyfile(Path(__file__), sources / Path(__file__).name)
    downloads = []
    try:
        for relative in FILES:
            if time.perf_counter() - started > 140:
                raise TimeoutError("The bounded source audit expired")
            path = sources / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.perf_counter()
            url = PREFIX + relative
            with urllib.request.urlopen(url, timeout=12) as response:
                data = response.read(200001)
            if not 100 < len(data) <= 200000:
                raise AssertionError("Official source size is outside the registered bound")
            content = data.decode("utf-8")
            if "Copyright (c), ETH Zurich and UNC Chapel Hill." not in content or "Redistribution and use in source and binary forms" not in content:
                raise AssertionError("The original copyright and license must remain intact")
            with path.open("xb") as handle:
                handle.write(data)
            downloads.append({"url": url, "source": data_identity(path),
                "lines": len(content.splitlines()), "seconds": time.perf_counter() - stamp,
                "original_copyright_and_license_preserved": True})
            save(output / ("FETCH_" + str(len(downloads)) + ".json"), downloads[-1])
        checks = {
            "all_nine_required_sources_from_fixed_official_commit": len(downloads) == len(FILES) == 9,
            "all_original_copyright_and_licenses_preserved": all(row["original_copyright_and_license_preserved"] for row in downloads),
            "audit_source_identity_unchanged": data_identity(Path(__file__)) == original,
            "private_source_output_below2GiB_and_disk20GiB": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 2 * 1024**3 and shutil.disk_usage(str(ROOT)).free >= 20 * 1024**3,
        }
        result = {"status": "official_matching_sources_preserved_pending_text_audit0",
            "checks": checks, "downloads": downloads, "elapsed_seconds": time.perf_counter() - started,
            "writer_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
            "parent_count": 36, "new_optimization_count": 0,
            "no_native_cache_misses_speed_or_quality_result_in_this_receipt": True}
        save(output / "RESULT.json", result)
        print(json.dumps({"checks": checks, "sources": len(downloads), "seconds": result["elapsed_seconds"], "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The official source audit failed verification")
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "downloads_completed": downloads,
            "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()

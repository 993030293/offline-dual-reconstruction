"""Encode all full public members without changing the original worker limit."""

import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np

from strict_sift_encoding import StrictSIFTEncoder, receipt, ROOT, BUILD
from audit_strict_sift_encoding_019 import array_identity
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/R171/g0"


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).parent
    paths = [Path(__file__), *[code / name for name in ("strict_sift_encoding.py", "native_sift_encoding.cpp",
        "integer_sift_matching.py", "audit_strict_sift_encoding_019.py", "localize_strict_encoding.py",
        "strict_encoding_node.py", "strict_encoding_owned_json_write.py", "codec_localization_node.py")],
        BUILD / "native_sift_encoding.dll"]
    sources = [data_identity(path) for path in paths]
    archive = OUT / "source_snapshot"
    archive.mkdir()
    for number, path in enumerate(paths):
        shutil.copyfile(path, archive / (f"{number:02d}_" + path.name))
    database = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/database.db"
    database_id = data_identity(database)
    a, b = StrictSIFTEncoder(False), StrictSIFTEncoder(True)
    rows = []
    with sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True) as connection:
        cursor = connection.execute("SELECT image_id,rows,cols,data FROM descriptors ORDER BY image_id")
        for image_id, count, columns, data in cursor:
            if columns != 128 or count <= 0 or len(data) != count*columns:
                raise ValueError("An actual full public descriptor member changed")
            original = np.frombuffer(data, np.uint8).reshape(count, columns)
            value = original.astype(np.float32)/512
            before = array_identity(value)
            left, right = a(value), b(value)
            rows.append({"image_id": image_id, "full_rows": count, "equal_layout_bits_ownership": array_identity(left) == array_identity(right),
                         "both_full_original_bytes_exact": left.tobytes() == right.tobytes() == data,
                         "source_array_unchanged": before == array_identity(value), "route": b.calls[-1]["route"]})
    checks = {"all360_original_members_complete_bits_layouts_inputs": len(rows) == 360
            and all(row["equal_layout_bits_ownership"] and row["both_full_original_bytes_exact"] and row["source_array_unchanged"] for row in rows),
        "235_supported_and125_original_large_row_routes": sum(row["route"] == "native_single_scan" for row in rows) == 235
            and sum(row["route"] == "original" for row in rows) == 125,
        "all_sources_compiled_library_current": sources == [data_identity(path) for path in paths],
        "original_public_database_unchanged": database_id == data_identity(database),
        "CPU_only_no_Torch_or_CUDA": "torch" not in sys.modules}
    result = {"status": "all360_full_public_CPU_encoding_passed_pending_full_localization0" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": sources, "database": database_id, "rows": rows,
        "raw": receipt(a), "native": receipt(b), "extra_actual_array_output_bytes": receipt(a)["actual_recorded_successful_output_bytes"]
            + receipt(b)["actual_recorded_successful_output_bytes"], "complete_seconds": time.perf_counter()-started,
        "not_matching_worker_support_for_125_large_rows_not_GT_or_unknown_object_training": True, "new_optimization_count": 0}
    atomic_json(OUT / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "extra_actual_array_output_bytes", "complete_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An original public encoding or scope guard failed")


if __name__ == "__main__":
    main()

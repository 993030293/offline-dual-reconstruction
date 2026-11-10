"""Package complete unchanged releases with negotiated lossless asset transport."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def byte_equal(left, right):
    while True:
        a, b = left.read(1024 * 1024), right.read(1024 * 1024)
        if a != b:
            return False
        if not a:
            return True


def build(source, destination, compressed, source_rows):
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for relative, identity in source_rows.items():
        original = source / relative
        packed = compressed and original.suffix == ".spk"
        filename = relative + ".gz" if packed else relative
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if packed:
            with original.open("rb") as src, target.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as encoded:
                    shutil.copyfileobj(src, encoded, length=1024 * 1024)
        else:
            shutil.copyfile(original, target)
        artifact = data_identity(target)
        with original.open("rb") as src:
            with (gzip.open(target, "rb") if packed else target.open("rb")) as decoded:
                if not byte_equal(src, decoded):
                    raise ValueError("A complete release file changed during packaging: " + relative)
        if data_identity(original) != identity:
            raise ValueError("A complete source release changed during packaging")
        rows.append({"relative": relative, "served_file": filename,
            "encoding": "gzip" if packed else "identity", "decoded_identity": identity,
            "encoded_identity": artifact, "independent_full_decoded_bytes_exact": True})
    catalog = {"schema_version": 1, "codec": "stdlib_gzip_level1_mtime0" if compressed else "identity",
        "source": str(source), "files": rows, "all_original_release_files_retained": True}
    atomic_json(destination / "FILE_INDEX.json", catalog)
    total_bytes = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    assets = [row for row in rows if Path(row["relative"]).suffix == ".spk"]
    index_bytes = (destination / "FILE_INDEX.json").stat().st_size
    return {"destination": str(destination), "compressed": compressed,
        "complete_first_build_and_full_verification_seconds": time.perf_counter() - started,
        "release_bytes_including_index": total_bytes, "index_bytes": index_bytes,
        "model_asset_bytes_including_required_index": sum(row["encoded_identity"]["bytes"] for row in assets) + index_bytes,
        "asset_count": len(assets), "files": len(rows), "all_original_files_exact_after_decoding": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    source.relative_to(ROOT / "04_outputs")
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005")
    if output.exists():
        raise FileExistsError("A fresh independent package campaign is required")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in source.rglob("*")) or not any(p.suffix == ".spk" for p in files):
        raise ValueError("Only a complete local native release without symlinks is admitted")
    source_size = sum(p.stat().st_size for p in files)
    if source_size * 6 > 30 * 1024**3 or shutil.disk_usage(ROOT / "04_outputs").free < source_size * 6 + 20 * 1024**3:
        raise MemoryError("The complete three-pair package exceeds its disk admission")
    output.mkdir(parents=True)
    paths = [Path(__file__), ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/runtime.py"]
    source_code = [data_identity(p) for p in paths]
    source_rows = {p.relative_to(source).as_posix(): data_identity(p) for p in files}
    atomic_json(output / "PROTOCOL.json", {"candidate": "R109_HTTP_LOSSLESS_ASSET_DELIVERY",
        "source": str(source), "source_rows": source_rows, "sources": source_code,
        "three_complete_paired_builds": True, "minimum_asset_plus_manifest_storage_reduction": .1,
        "complete_release_occupancy_reported_separately": True,
        "compression_build_cost_is_not_navigation_cost_or_claimed_speed": True,
        "actual_browser_navigation_guards_and_controls_pending": True, "counted": False})
    pairs = []
    for trial in range(3):
        row = {"trial": trial, "modes": {}}
        for mode in (("identity", "gzip") if trial % 2 == 0 else ("gzip", "identity")):
            row["modes"][mode] = build(source, output / f"trial{trial}" / mode, mode == "gzip", source_rows)
        original, candidate = row["modes"]["identity"], row["modes"]["gzip"]
        row["asset_plus_index_reduction"] = 1 - candidate["model_asset_bytes_including_required_index"] / original["model_asset_bytes_including_required_index"]
        row["full_release_occupancy_reduction"] = 1 - candidate["release_bytes_including_index"] / original["release_bytes_including_index"]
        row["all_complete_source_rows_current"] = source_rows == {p.relative_to(source).as_posix(): data_identity(p) for p in files}
        pairs.append(row)
        atomic_json(output / "PROGRESS.json", {"pairs": pairs, "counted": False})
        print(json.dumps(row), flush=True)
        if not row["all_complete_source_rows_current"]:
            raise ValueError("A full source identity changed before completing the pair")
    passed = all(p["asset_plus_index_reduction"] >= .1 and p["all_complete_source_rows_current"] for p in pairs)
    result = {"status": "storage_passed_browser_pending" if passed else "storage_gate_failed",
        "pairs": pairs, "asset_plus_index_reduction_median": statistics.median(p["asset_plus_index_reduction"] for p in pairs),
        "full_release_occupancy_reduction_median": statistics.median(p["full_release_occupancy_reduction"] for p in pairs),
        "first_build_cost_medians": {m: statistics.median(p["modes"][m]["complete_first_build_and_full_verification_seconds"] for p in pairs)
            for m in ("identity", "gzip")}, "source_code_current": source_code == [data_identity(p) for p in paths],
        "new_training_or_math_changes": False, "counted": False}
    atomic_json(output / "RESULT.json", result)


if __name__ == "__main__":
    main()

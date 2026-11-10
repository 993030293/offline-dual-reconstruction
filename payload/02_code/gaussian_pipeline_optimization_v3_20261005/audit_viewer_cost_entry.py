"""Locate the current preview entry and existing mechanisms without benchmarking it."""

import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/viewer_cost_005"
RELEASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release"
PAGE = RELEASE / "baseline/cases/packed"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OUT)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), PAGE / "viewer.js", PAGE / "index.html", PAGE / "render_schedule.js", PAGE / "lazy_models.js",
        PAGE / "motion_resolution.js", PAGE / "packed_splat_asset.js", RELEASE / "shared-build/vendor/spark.module.js",
        RELEASE / "baseline/ASSET_MANIFEST.json"]
    identities = [data_identity(path) for path in paths]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    viewer, scheduler, lazy, motion, spark = texts[1], texts[3], texts[4], texts[5], texts[7]
    prefix = "var jsContent = "
    worker_literals = [line[len(prefix):-1] for line in spark.splitlines() if line.startswith(prefix) and line.endswith(";")]
    if len(worker_literals) != 1:
        raise ValueError("The original embedded worker declaration changed")
    literal = worker_literals[0]
    if any(ord(value) < 32 and value != "\t" for value in literal):
        raise ValueError("The native embedded worker has an unexpected raw control character")
    # JavaScript string literals permit raw horizontal tabs that strict JSON rejects.
    worker = json.loads(literal, strict=False)
    facts = {
        "existing_demand_frame_scheduler": 'state.renderPolicy = "demand"' in viewer and "if (scheduled) return" in scheduler,
        "existing_selected_only_lazy_model_loading": 'state.modelLoadingPolicy = "selected_only"' in viewer and "if (pending.has(mode)) return pending.get(mode)" in lazy,
        "existing_motion_resolution_and_stop_restore": 'fullRatio * 0.75' in motion and 'moving ? movingRatio : fullRatio' in motion,
        "existing_free_flight_and_drone_controls": all(value in viewer for value in ('"KeyW"', '"KeyQ"', 'name === "drone"', 'name === "top"', 'state.navigationMode === "fly"')),
        "existing_worker_sort32_dispatch": 'this.sortWorker.call("sortSplats32"' in spark and "sortSplats32" in worker,
        "existing_worker_pool_capacity4": "constructor(maxWorkers = 4)" in spark,
        "existing_fixed_model_no_LOD_requested": 'lod: false' in viewer,
        "existing_original_model_asset_and_vendor_validation": 'verifySparkBuild(' in viewer and 'loadPackedAsset(' in viewer,
    }
    checks = {"all_listed_existing_source_mechanisms_located": all(facts.values()),
        "all_original_sources_and_asset_manifest_unchanged": identities == [data_identity(path) for path in paths]}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        (snapshot / (f"{index:03d}_" + path.name)).write_bytes(path.read_bytes())
    lines = {value: viewer[:viewer.index(value)].count("\n") + 1 for value in (
        'state.renderPolicy = "demand"', 'state.modelLoadingPolicy = "selected_only"', 'renderSchedule.start(', 'renderer.render(scene, camera)')}
    worker_lines = [index + 1 for index, line in enumerate(worker.splitlines()) if "sortSplats32" in line and len(line) < 500]
    atomic_json(output / "RESULT.json", {"status": "current_preview_source_contract_and_existing_mechanisms_located0" if all(checks.values()) else "failed_source_verification",
        "checks": checks, "source_facts": facts, "source_lines": lines,
        "embedded_worker_sort32_lines": worker_lines, "worker_decoded_as_native_text_not_executed": True,
        "sources": identities, "entry": str(PAGE / "index.html"),
        "browser_GPU_or_server_live_state": "not_verified_no_browser_or_service_started",
        "performance_new_algorithms_quality_recovery": "not_run", "new_optimization_count": 0,
        "parent_accepted_count": 33, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"checks": checks, "source_facts": facts, "source_lines": lines, "new_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The current preview entry differs from the audited source boundary")


if __name__ == "__main__":
    main()

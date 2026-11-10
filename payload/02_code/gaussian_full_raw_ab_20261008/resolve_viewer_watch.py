"""Resolve one owned idle watcher after all source pipelines have terminal outcomes."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import psutil

from atomic_json_windows import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--watch", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    args.watch.resolve().relative_to(campaign)
    state = read_json(args.watch / "STATE.json")
    if state["status"] != "running":
        raise ValueError("Only the existing idle watcher may be resolved")
    sources = []
    for case in read_json(campaign / "CASES.json")["cases"]:
        for mode in ("baseline", "optimized"):
            side = campaign / case["case"] / mode
            outcome = read_json(side / "PIPELINE_STATE.json")
            if outcome["status"] == "running":
                raise RuntimeError("A source pipeline still needs its healthy watcher")
            train_path = side / "training/STATE.json"
            trained = read_json(train_path) if train_path.is_file() else None
            branch = case["case"] + "-" + mode
            if trained and trained["status"] == "completed":
                manifest_path = campaign / "viewer" / branch / "ASSET_MANIFEST.json"
                manifest = read_json(manifest_path)
                if manifest["scene"]["numSplats"] != trained["gaussian_count"]:
                    raise ValueError("The staged viewer population differs from completed training")
                sources.append({"branch": branch, "status": "complete_model_and_native_packed_assets",
                    "manifest": str(manifest_path)})
            elif str(outcome["status"]).startswith("failed"):
                sources.append({"branch": branch, "status": "source_pipeline_failed_no_model",
                    "pipeline": str(side / "PIPELINE_STATE.json")})
            else:
                raise RuntimeError("A source outcome remains unresolved")
    if len(sources) != 8 or sum(row["status"] == "complete_model_and_native_packed_assets" for row in sources) != 5:
        raise ValueError("This watcher resolution must preserve the observed five-model, three-failure outcomes")
    process = psutil.Process(state["pid"])
    expected_script = Path(__file__).with_name("run_viewer_build_watch.py").resolve()
    argv = process.cmdline()
    if abs(process.create_time()-state["process_birth"]) > .001:
        raise RuntimeError("The watcher PID was reused")
    if len(argv) != 4 or Path(argv[1]).resolve() != expected_script or argv[2] != "--campaign" or Path(argv[3]).resolve() != campaign:
        raise RuntimeError("The target is not this exact campaign watcher")
    if process.children(recursive=True):
        raise RuntimeError("The watcher has live native children and must not be stopped")
    atomic_json(args.watch / "SOURCE_STATE_BEFORE_STOP.json", state)
    process.terminate()
    process.wait(timeout=10)
    result = {"status": "resolved_five_packed_models_three_source_failures", "watcher_identity": {
        "pid": state["pid"], "process_birth": state["process_birth"], "argv": argv},
        "resolved_UTC": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "original_state_preserved": str(args.watch / "SOURCE_STATE_BEFORE_STOP.json"),
        "stopped_idle_owned_watcher_only": True, "preview_HTTP_server_retained": True}
    atomic_json(args.watch / "RESOLUTION.json", result)
    atomic_json(args.watch / "STATE.json", result)


if __name__ == "__main__":
    main()

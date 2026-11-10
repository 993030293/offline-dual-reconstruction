"""Run three fresh ordered native Scene pairs under the existing exclusive lease."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from mapped_image_node_config import ROOT, CODE, OUT, atomic_json, data_identity, read
from compare_mapped_scene import compare_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    first = compare_pair(OUT/"p1", OUT/"np1n", OUT/"np1m")
    if first["status"] != "complete_Scene_pair_exact_memory_screen_passed":
        raise ValueError("The complete diagnostic first-build pair did not pass")
    paths = [Path(__file__), CODE/"compare_mapped_scene.py", CODE/"mapped_training_images.py",
        CODE/"profile_mapped_scene.py", CODE/"mapped_image_node.py", CODE/"run_with_resource_lease.py"]
    sources = [data_identity(path) for path in paths]
    started = time.perf_counter()
    state = {"status": "running", "sources": sources, "first_build_diagnostic": first,
        "orders": [["native", "mapped"], ["mapped", "native"], ["native", "mapped"]],
        "private_cache": str(OUT/"full_cache_001"), "pairs": [], "new_optimization_count": 0}
    atomic_json(args.output/"STATE.json", state)
    for index, order in enumerate(state["orders"]):
        pair = args.output/f"pair_{index:03d}"
        pair.mkdir()
        nodes = {}
        for mode in order:
            output = pair/mode
            node = pair/(mode+"_node")
            lease = pair/(mode+"_lease.json")
            nodes[mode] = node
            mounted = "/mnt/c/"+str(output.relative_to(Path("C:/"))).replace("\\", "/")
            child = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_mapped_scene.py",
                "--mode", "baseline" if mode == "native" else "mapped", "--output", mounted]
            if mode == "mapped":
                child += ["--cache", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001/full_cache_001"]
            command = [sys.executable, str(CODE/"run_with_resource_lease.py"), "--owner", f"root_R131_Scene_{index}_{mode}_single_chat",
                "--receipt", str(lease), "--", sys.executable, str(CODE/"mapped_image_node.py"), "--kind", "camera_oracle",
                "--mode", "diagnostic", "--node", str(node), "--output", str(output), "--", *child]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                state.update(status="failed_original_node", failed_pair=index, failed_mode=mode,
                    complete_campaign_seconds=time.perf_counter()-started)
                atomic_json(args.output/"STATE.json", state)
                raise RuntimeError("An original full Scene node failed; no restart or accepted claim")
        result = compare_pair(pair, nodes["native"], nodes["mapped"])
        state["pairs"].append(result)
        atomic_json(args.output/"STATE.json", state)
        if result["status"] != "complete_Scene_pair_exact_memory_screen_passed":
            raise AssertionError("A full Scene pair lost scientific equivalence or its original memory gate")
    native = [row["actual_process_tree_peak_RSS_bytes"]["native"] for row in state["pairs"]]
    mapped = [max(row["actual_process_tree_peak_RSS_bytes"]["mapped"], first["actual_process_tree_peak_RSS_bytes"]["mapped"]) for row in state["pairs"]]
    medians = {"native": statistics.median(native), "mapped_conservative_including_first_build": statistics.median(mapped)}
    reduction = 1-medians["mapped_conservative_including_first_build"]/medians["native"]
    state.update(status="three_complete_Scene_pairs_passed_training_quality_recovery_pending",
        three_pair_conservative_peak_RSS_medians=medians, three_pair_peak_RSS_reduction=reduction,
        complete_campaign_seconds=time.perf_counter()-started,
        sources_unchanged=sources == [data_identity(path) for path in paths])
    if not state["sources_unchanged"] or reduction < .2:
        raise AssertionError("The full Scene campaign lost its declared source or primary memory gate")
    atomic_json(args.output/"RESULT.json", state)
    atomic_json(args.output/"STATE.json", state)
    print(json.dumps({"status": state["status"], "three_pair_peak_RSS_reduction": reduction, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()

"""Compare native matching on identical fresh features, including shard preparation and merge."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V3)]
from runtime import data_identity
from atomic_json_windows import atomic_json
from native_sfm_match_shards import partition_pairs, merge_databases, compare_databases
from gpu_lease import gpu_lease


def native_match(commands, folder, maximum_seconds=1800):
    children = []
    started = time.monotonic()
    with ExitStack() as stack:
        try:
            for index, command in enumerate(commands):
                log = stack.enter_context((folder / ("native_" + str(index) + ".log")).open("x"))
                children.append(subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT))
            while any(child.poll() is None for child in children):
                if time.monotonic()-started > maximum_seconds:
                    raise TimeoutError("The fixed native matching compute budget expired")
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("A native matcher failed")
                time.sleep(.2)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("A native matcher returned an error")
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()
    return time.monotonic()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-isolated", action="store_true")
    args = parser.parse_args()
    if args.repeats != 3:
        raise ValueError("The paired protocol requires three complete native repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [data_identity(args.features), data_identity(args.pairs)]
    with sqlite3.connect(args.features.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        if any(connection.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
               for name in ("matches", "two_view_geometries")):
            raise ValueError("A same-feature ablation requires an unmatched fresh feature database")
    pair_lines = args.pairs.read_text().splitlines()
    binary = ROOT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    result = {"status": "running", "mechanism": "R165", "sources": sources,
        "source": data_identity(Path(__file__)), "native_binary": data_identity(binary),
        "all_original_pairs": len(pair_lines), "same_feature_IDs_and_columns": True,
        "OS_page_cache_cleared": False,
        "resource_scope": "GPU serial lease; owned primary and localization work drained" if args.require_isolated else
            "GPU serial lease; possible CPU overlap is an engineering timing limitation",
        "unrelated_system_activity_excluded": False,
        "whole_pipeline_speed_claim": False, "trials": [], "complete_oracles": []}
    atomic_json(args.output / "CONTRACT.json", result)
    try:
        for repeat in range(3):
            databases = {}
            for mode in (("baseline", "optimized") if repeat != 1 else ("optimized", "baseline")):
                if args.require_isolated:
                    from run_isolated_matching_after_audits import blockers
                    if blockers(args.features.resolve().parents[3]):
                        raise RuntimeError("Owned primary or localization compute resumed during the isolated comparison")
                folder = args.output / ("pair" + str(repeat)) / mode
                folder.mkdir(parents=True)
                with gpu_lease(folder / "GPU_LEASE.json", "R165_" + str(repeat) + "_" + mode) as lease:
                    started = time.monotonic()
                    database = folder / "database.db"
                    shutil.copyfile(args.features, database)
                    paths = [(database, args.pairs)]
                    partition = None
                    if mode == "optimized":
                        subsets, partition = partition_pairs(database, pair_lines)
                        paths = []
                        for number, subset in enumerate(subsets):
                            path = database if number == 0 else folder / "shard1.db"
                            if number:
                                shutil.copyfile(args.features, path)
                            pairs = folder / ("pairs" + str(number) + ".txt")
                            pairs.write_text("\n".join(subset)+"\n")
                            paths.append((path, pairs))
                    commands = [[binary, "matches_importer", "--database_path", database_path,
                        "--match_list_path", pairs_path, "--match_type", "pairs", "--FeatureMatching.use_gpu", "1",
                        "--FeatureMatching.gpu_index", "0", "--FeatureMatching.num_threads", "8",
                        "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"]
                        for database_path, pairs_path in paths]
                    native_seconds = native_match(commands, folder)
                    merged = merge_databases(database, paths[1][0]) if len(paths) == 2 else None
                    cost = time.monotonic()-started
                result["trials"].append({"pair": repeat, "mode": mode, "preparation_native_merge_seconds": cost,
                    "native_seconds": native_seconds, "GPU_queue_wait_seconds": lease["wait_seconds"],
                    "partition": partition, "merge": merged, "database": str(database)})
                databases[mode] = database
                atomic_json(args.output / "RESULT.json", result)
            result["complete_oracles"].append(compare_databases(databases["baseline"], databases["optimized"]))
            atomic_json(args.output / "RESULT.json", result)
        if sources != [data_identity(args.features), data_identity(args.pairs)]:
            raise ValueError("The preserved fresh feature inputs changed")
        medians = {mode: float(np.median([row["preparation_native_merge_seconds"] for row in result["trials"]
                                        if row["mode"] == mode])) for mode in ("baseline", "optimized")}
        result.update(status="complete_native_columns_equal_with_observed_stage_benefit" if medians["optimized"] < medians["baseline"]
                      else "complete_native_columns_equal_no_observed_stage_speed_benefit",
                      median_seconds=medians, reduction_fraction=1-medians["optimized"]/medians["baseline"],
                      input_unchanged=True)
    except BaseException as error:
        result.update(status="failed_complete_native_matching_oracle", error=repr(error))
        atomic_json(args.output / "RESULT.json", result)
        raise
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "median_seconds", "reduction_fraction")}))


if __name__ == "__main__":
    main()

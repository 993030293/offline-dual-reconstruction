"""Run the original metric branches in isolated owned native processes."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
COMMON = ROOT / "02_code/box2_lio_guided_sfm_v1_20261002"
sys.path[:0] = [str(V1), str(COMMON)]
from runtime import atomic_json, data_identity

ORIGINAL = V1 / "run_metric_refinement.py"
LABELS = ("visual_metric_aligned", "weak_position_prior")


def prepare_leaf(label):
    if label not in LABELS:
        raise ValueError("Unknown original metric branch")
    source = ORIGINAL.read_text(encoding="utf-8")
    before = '    for label in ["visual_metric_aligned", "weak_position_prior"]:\n'
    if source.count(before) != 1:
        raise ValueError("Original branch loop changed")
    source = source.replace(before, "    for label in " + repr([label]) + ":\n")
    before = "        folder = args.output/label\n"
    if source.count(before) != 1:
        raise ValueError("Original branch output routing changed")
    source = source.replace(before, "        folder = args.output.parent.parent/label\n")
    marker = '    if any(case.get("status") == "failed_projection_validation" for case in results.values()):\n'
    if source.count(marker) != 1:
        raise ValueError("Original metric join changed")
    source = source[:source.index(marker)] + '''    leaf = results[label]
    atomic_json(args.output/"LEAF_RESULT.json", common.serializable({
        "label": label, "result": leaf, "elapsed_seconds": time.monotonic()-started,
        "independent_branch_only": True, "counted": False}))
    if leaf.get("status") == "failed_projection_validation":
        raise ValueError("Original branch has invalid observed projections")
    atomic_json(args.output/"STATE.json", {"status": "leaf_completed", "label": label})
'''
    return compile(source, str(ORIGINAL), "exec")


@contextmanager
def capture_native_configuration(output):
    import numpy as np
    import pycolmap
    import common
    factories = (pycolmap.create_default_ceres_bundle_adjuster,
                 pycolmap.create_pose_prior_ceres_bundle_adjuster)

    def capture(options, config, model, priors=None, prior_options=None):
        label = LABELS[0] if priors is None else LABELS[1]
        expected = common.ba_options(60)
        expected.ceres.solver_options.max_solver_time_in_seconds = 180
        expected.ceres.solver_options.num_threads = 8
        actual = common.serializable(options.todict())
        if actual != common.serializable(expected.todict()):
            raise AssertionError("Original complete solver options changed")
        if set(config.images) != {im.image_id for im in model.images.values() if im.has_pose}:
            raise AssertionError("Original configured images changed")
        if priors is None and config.fixed_gauge != pycolmap.BundleAdjustmentGauge.THREE_POINTS:
            raise AssertionError("Original visual gauge changed")
        if priors is not None:
            if (prior_options.prior_position_fallback_stddev != .15
                    or prior_options.ceres.prior_position_loss_function_type != pycolmap.LossFunctionType.CAUCHY
                    or prior_options.ceres.prior_position_loss_scale != 2.7954834829151074
                    or any(not np.array_equal(p.position_covariance, np.eye(3)*.15**2) for p in priors)):
                raise AssertionError("Original complete prior options changed")
        folder = output / (label + "_INITIAL_MODEL")
        folder.mkdir()
        model.write(folder)
        atomic_json(output / (label + "_EFFECTIVE_OPTIONS.json"), {
            "label": label, "solver_options": actual, "gauge": config.fixed_gauge.name,
            "configured_images": sorted(config.images), "visual_residuals": int(config.num_residuals(model)),
            "prior_count": 0 if priors is None else len(priors),
            "prior_options": None if prior_options is None else common.serializable(prior_options.todict()),
            "native_factory_unmodified": True})

    def visual(options, config, model):
        capture(options, config, model)
        return factories[0](options, config, model)

    def weak(options, prior_options, config, priors, model):
        capture(options, config, model, priors, prior_options)
        return factories[1](options, prior_options, config, priors, model)

    pycolmap.create_default_ceres_bundle_adjuster = visual
    pycolmap.create_pose_prior_ceres_bundle_adjuster = weak
    try:
        yield
    finally:
        pycolmap.create_default_ceres_bundle_adjuster = factories[0]
        pycolmap.create_pose_prior_ceres_bundle_adjuster = factories[1]


def run_original(arguments, branch=None):
    namespace = {"__file__": str(ORIGINAL), "__name__": "r119_original_private"}
    code = prepare_leaf(branch) if branch else compile(ORIGINAL.read_text(encoding="utf-8"), str(ORIGINAL), "exec")
    exec(code, namespace)
    previous = sys.argv
    sys.argv = [str(ORIGINAL)] + arguments
    output = Path(arguments[arguments.index("--output") + 1])
    try:
        with capture_native_configuration(output):
            namespace["main"]()
    finally:
        sys.argv = previous


def owned_tree(process, identities):
    try:
        parent = psutil.Process(process.pid)
        for item in [parent] + parent.children(recursive=True):
            identities[item.pid] = item.create_time()
    except psutil.NoSuchProcess:
        pass


def alive_owned(identities):
    alive = []
    for pid, born in identities.items():
        try:
            item = psutil.Process(pid)
            if item.create_time() == born and item.is_running():
                alive.append(item)
        except psutil.NoSuchProcess:
            pass
    return alive


def join_results(output, leaves, started):
    results = {}
    for label in LABELS:
        receipt = json.loads((leaves[label] / "LEAF_RESULT.json").read_text(encoding="utf-8"))
        if receipt["label"] != label:
            raise ValueError("Original branch result identity changed")
        results[label] = receipt["result"]
        if results[label].get("status") == "failed_projection_validation":
            raise ValueError("One or both original metric branches have invalid projections")
    contracts = [json.loads((leaves[label] / "CONTRACT.json").read_text(encoding="utf-8")) for label in LABELS]
    if contracts[0] != contracts[1]:
        raise AssertionError("The two branches did not share the complete original contract")
    alignments = [json.loads((leaves[label] / "ALIGNMENT.json").read_text(encoding="utf-8")) for label in LABELS]
    if alignments[0] != alignments[1]:
        raise AssertionError("The two native initial Sim3 alignments differ")
    delta = (results[LABELS[1]]["metrics"]["reprojection_px"]["p95"]
             - results[LABELS[0]]["metrics"]["reprojection_px"]["p95"])
    selected = LABELS[1] if delta <= .05 else LABELS[0]
    atomic_json(output / "RESULT.json", {
        "status": "metric_interface_verified", "results": results,
        "common_observation_p95_delta_px": delta, "development_guard_passed": delta <= .05,
        "selected": selected, "metric_source": "recorded_odometry_meters_not_GT",
        "elapsed_seconds": time.monotonic()-started, "absolute_accuracy_claim": False, "counted": False,
        "fallback_boundary": "If weak BA fails its guard, retain the separately identified post-hoc metric alignment; do not claim active metric BA success"})
    atomic_json(output / "CONTRACT.json", contracts[0])
    atomic_json(output / "ALIGNMENT.json", alignments[0])
    atomic_json(output / "STATE.json", {"status": "completed", "selected": selected,
        "registered_images": contracts[0]["input_camera_count"],
        "supported_priors": contracts[0]["supported_position_prior_count"],
        "visual_only_pose_gap_images": contracts[0]["pose_gap_images_without_priors"],
        "development_guard_passed": delta <= .05})


def run_dag(arguments):
    output = Path(arguments[arguments.index("--output") + 1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    branch_root = output / "branches"
    branch_root.mkdir()
    started = time.monotonic()
    processes, streams, identities, intervals, leaves = {}, {}, {}, {}, {}
    peak = 0
    try:
        for label in LABELS:
            leaf = branch_root / label
            leaves[label] = leaf
            leaf_arguments = list(arguments)
            leaf_arguments[leaf_arguments.index("--output") + 1] = str(leaf)
            streams[label] = (output / (label + ".log")).open("x", encoding="utf-8")
            command = [sys.executable, "-B", str(Path(__file__)), "--leaf", label, "--"] + leaf_arguments
            process = subprocess.Popen(command, stdout=streams[label], stderr=subprocess.STDOUT)
            processes[label] = process
            intervals[label] = {"started_monotonic": time.monotonic(), "command": command}
            owned_tree(process, identities)
        while any(p.poll() is None for p in processes.values()):
            for label, process in processes.items():
                owned_tree(process, identities)
                if process.poll() is not None and "ended_monotonic" not in intervals[label]:
                    intervals[label]["ended_monotonic"] = time.monotonic()
            current = 0
            for item in alive_owned(identities):
                try:
                    current += item.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, current)
            atomic_json(output / "DAG_STATE.json", {"status": "running", "elapsed_seconds": time.monotonic()-started,
                "branch_intervals": intervals, "peak_owned_RSS_bytes": peak,
                "owned_process_identities": [{"pid": p, "creation_time": c} for p, c in identities.items()]})
            if time.monotonic()-started > 1800 or current > 8*1024**3:
                raise RuntimeError("The preregistered DAG time or owned RSS bound was exceeded")
            if any(p.poll() not in (None, 0) for p in processes.values()):
                raise RuntimeError("An owned original metric branch failed; preserve unaccepted partial outputs")
            time.sleep(.2)
        for label, process in processes.items():
            process.wait()
            intervals[label].setdefault("ended_monotonic", time.monotonic())
            intervals[label]["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError("An owned original metric branch failed")
        join_results(output, leaves, started)
    finally:
        cancelled = []
        for item in reversed(alive_owned(identities)):
            item.kill()
            cancelled.append(item)
        psutil.wait_procs(cancelled, timeout=10)
        for process in processes.values():
            if process.poll() is None:
                process.kill()
            process.wait()
        for stream in streams.values():
            stream.close()
        close = {"owned_process_identities": [{"pid": p, "creation_time": c} for p, c in identities.items()],
            "alive_owned_processes": [{"pid": p.pid, "creation_time": p.create_time()} for p in alive_owned(identities)],
            "cancellation_kills": len(cancelled), "branch_intervals": intervals,
            "peak_owned_RSS_bytes": peak, "elapsed_seconds": time.monotonic()-started, "counted": False}
        atomic_json(output / "DAG_RESOURCE_CLOSE.json", close)
        if close["alive_owned_processes"]:
            raise RuntimeError("An owned native branch did not close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf", choices=LABELS)
    parser.add_argument("--mode", choices=("serial", "dag"))
    parser.add_argument("--source-preparation-only", action="store_true")
    parser.add_argument("original_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.original_arguments[1:] if args.original_arguments[:1] == ["--"] else args.original_arguments
    if args.source_preparation_only:
        for label in LABELS:
            prepare_leaf(label)
        print(json.dumps({"both_source_bindings_prepared": True, "native_solvers_executed": False}))
    elif args.leaf:
        run_original(arguments, args.leaf)
    elif args.mode == "serial":
        run_original(arguments)
    elif args.mode == "dag":
        run_dag(arguments)
    else:
        parser.error("Choose a private original leaf, serial control or DAG candidate")


if __name__ == "__main__":
    main()

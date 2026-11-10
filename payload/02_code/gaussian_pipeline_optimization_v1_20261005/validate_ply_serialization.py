"""Compare packed PLY serialization with native bytes and isolated save costs."""

import argparse
import gc
import json
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

from runtime import activate_reference, atomic_json, data_identity
from ply_serialization import structured_vertices, vectorized_ply_model


def numeric(output):
    from plyfile import PlyData, PlyElement

    checks = {}
    rng = np.random.default_rng(20261005)
    for degree in (0, 3):
        names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
        names += [f"f_rest_{i}" for i in range(3*((degree+1)**2-1))]
        names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        for size in (0, 1, 127):
            matrix = rng.normal(size=(size, len(names)*2)).astype(np.float32)[:, ::2]
            if size:
                matrix[0, :6] = [0., -0., np.finfo(np.float32).tiny, np.finfo(np.float32).max, -7., 3.]
            old = np.empty(size, dtype=[(name, "f4") for name in names])
            old[:] = list(map(tuple, matrix))
            new = structured_vertices(matrix, names)
            prefix = f"sh{degree}_n{size}"
            before, after = output/f"{prefix}_native.ply", output/f"{prefix}_vectorized.ply"
            PlyData([PlyElement.describe(old, "vertex")]).write(str(before))
            PlyData([PlyElement.describe(new, "vertex")]).write(str(after))
            loaded = PlyData.read(str(after))["vertex"].data
            checks[prefix] = (before.read_bytes() == after.read_bytes()
                              and loaded.dtype.names == old.dtype.names and loaded.tobytes() == old.tobytes())
    for label, matrix, names in [
            ("reject_float64", np.zeros((2, 1), dtype=np.float64), ["x"]),
            ("reject_shape", np.zeros((2, 2), dtype=np.float32), ["x"]),
            ("reject_duplicate", np.zeros((2, 2), dtype=np.float32), ["x", "x"])]:
        try:
            structured_vertices(matrix, names)
        except ValueError:
            checks[label] = True
        else:
            checks[label] = False
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = activate_reference(args.runtime)
    import torch
    from scene import GaussianModel

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    if budget <= 0:
        raise MemoryError("Insufficient GPU headroom")
    cls = GaussianModel if args.worker == "native" else vectorized_ply_model(GaussianModel)
    model = cls(3)
    model.load_ply(str(args.model))
    input_count = len(model.get_xyz)
    torch.cuda.synchronize()
    gc.collect()
    state = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
    started = time.monotonic()
    model.save_ply(str(args.output/"point_cloud.ply"))
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    peak_rss_save = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    rng_unchanged = (torch.equal(state[0], torch.get_rng_state())
        and all(torch.equal(a, b) for a, b in zip(state[1], torch.cuda.get_rng_state_all()))
        and state[2] == random.getstate()
        and state[3][0] == np.random.get_state()[0]
        and np.array_equal(state[3][1], np.random.get_state()[1])
        and state[3][2:] == np.random.get_state()[2:])
    reloaded = GaussianModel(3)
    reloaded.load_ply(str(args.output/"point_cloud.ply"))
    fields = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    exact = {name: bool(torch.equal(getattr(model, name), getattr(reloaded, name))) for name in fields}
    resource_ok = peak_rss_save <= 32*1024**3 and torch.cuda.max_memory_reserved() <= budget and elapsed <= 300
    result = {"status": "saved_verified" if all(exact.values()) and rng_unchanged and resource_ok else "failed",
        "variant": args.worker, "save_seconds": elapsed, "peak_rss_after_save_bytes": peak_rss_save,
        "maximum_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu_budget_bytes": budget,
        "input_vertices": input_count, "native_reload_exact": exact, "rng_unchanged": rng_unchanged,
        "resource_guard_passed": resource_ok, "file": data_identity(args.output/"point_cloud.ply"),
        "runtime": runtime, "source": data_identity(Path(__file__)),
        "adapter": data_identity(Path(__file__).with_name("ply_serialization.py"))}
    atomic_json(args.output/"RESULT.json", result)
    if result["status"] != "saved_verified":
        raise AssertionError("Serialization worker failed the frozen guards")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["native", "vectorized"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    free = shutil.disk_usage(args.output).free
    if free < 20*1024**3:
        raise OSError("Serialization experiments require at least20GiB disk headroom")
    before = data_identity(args.model)
    state = {"status": "numeric_checks", "input": before, "counted": False,
             "orders": [["native", "vectorized"], ["vectorized", "native"], ["native", "vectorized"]],
             "threshold_median_save_reduction": .1, "maximum_worker_seconds": 300,
             "disk_headroom_bytes": free, "new_group_disk_budget_bytes": 30*1024**3,
             "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("ply_serialization.py"))]}
    atomic_json(args.output/"STATE.json", state)
    numeric_dir = args.output/"numeric"
    numeric_dir.mkdir()
    checks = numeric(numeric_dir)
    atomic_json(numeric_dir/"RESULT.json", {"status": "passed", "checks": checks})
    rows = []
    for trial, order in enumerate(state["orders"]):
        trial_rows = {}
        for variant in order:
            output = args.output/f"trial{trial}_{variant}"
            command = [sys.executable, str(Path(__file__)), "--runtime", str(args.runtime), "--model", str(args.model),
                       "--output", str(output), "--worker", variant]
            state.update(status="saving", trial=trial, variant=variant, command=command)
            atomic_json(args.output/"STATE.json", state)
            with (args.output/f"trial{trial}_{variant}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Serialization worker failed: {output}")
            trial_rows[variant] = json.loads((output/"RESULT.json").read_text())
        same = trial_rows["native"]["file"]["sha256"] == trial_rows["vectorized"]["file"]["sha256"]
        if not same:
            raise AssertionError("Real native and vectorized file bytes differ")
        rows.append({"trial": trial, "byte_identical": same, **trial_rows})
    unchanged = data_identity(args.model) == before
    medians = {variant: float(np.median([row[variant]["save_seconds"] for row in rows]))
               for variant in ("native", "vectorized")}
    reduction = 1-medians["vectorized"]/medians["native"]
    state.update(status="stage_guards_passed_integration_pending" if reduction >= .1 and unchanged else "rejected",
                 numeric_checks=checks, rows=rows, median_save_seconds=medians, median_reduction=reduction,
                 input_unchanged=unchanged, real_training_entry_verified=False,
                 boundary="Buffered save API including D2H and serialization; no full-training speed claim")
    atomic_json(args.output/"STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "median_save_seconds", "median_reduction")}), flush=True)


if __name__ == "__main__":
    main()

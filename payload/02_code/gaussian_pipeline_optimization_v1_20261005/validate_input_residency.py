"""Measure native CPU versus CUDA camera residency without changing pixel values."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time

from runtime import activate_reference, atomic_json, data_identity


def fingerprint(tensor):
    if tensor is None:
        return None
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def numeric_checks():
    import torch
    values = torch.arange(256, dtype=torch.float32).reshape(1, 16, 16) / 255
    checks = {}
    for label, original in (("contiguous", values.repeat(3, 1, 1)),
                            ("transpose", values.repeat(3, 1, 1).transpose(1, 2)),
                            ("slice", values.repeat(3, 1, 1)[:, ::2, ::2]),
                            ("constant_alpha", torch.full((1, 16, 16), .123)),
                            ("pixel_alpha", values)):
        moved = original.to("cuda")
        checks[label + "_bits"] = torch.equal(original.contiguous().view(torch.int32), moved.cpu().contiguous().view(torch.int32))
        checks[label + "_shape_dtype"] = original.shape == moved.shape and original.dtype == moved.dtype
        if label != "slice":
            checks[label + "_dense_stride"] = original.stride() == moved.stride()
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if args.expected_image_count <= 0 or (args.input_version == "C212_frozen" and args.expected_image_count != 212):
        raise ValueError("A different image denominator requires an explicit new input version")
    image_paths = [p for p in (args.dataset/"images").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
    if len(image_paths) != args.expected_image_count:
        raise ValueError("The actual image count differs from the explicit capacity contract")
    metadata = activate_reference(args.runtime)
    import numpy as np
    import psutil
    import torch
    from arguments import ModelParams
    from scene import Scene, GaussianModel
    from full_intrinsics import camera_loader_adapter
    from host_memory import trim_unused_host_pages

    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    budget = min(12 * 1024**3, free - 2 * 1024**3)
    if budget <= 0:
        raise MemoryError("Insufficient GPU headroom")
    torch.cuda.reset_peak_memory_stats()
    parser = argparse.ArgumentParser()
    group = ModelParams(parser)
    native = parser.parse_args([])
    for key, value in json.loads((args.runtime / "MAIN_CONFIG.json").read_text())["parameters"].items():
        setattr(native, key, value)
    if native.data_device != "cpu" or native.train_test_exp or native.depths:
        raise ValueError("The frozen control must use plain CPU RGB inputs")
    native.data_device = args.worker
    native.source_path = str(args.dataset.resolve())
    native.model_path = str((args.output / "scene").resolve())
    Path(native.model_path).mkdir()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    started = time.monotonic()
    with camera_loader_adapter(args.dataset):
        scene = Scene(group.extract(native), GaussianModel(3), shuffle=False)
    torch.cuda.synchronize()
    trim = trim_unused_host_pages()
    load_seconds = time.monotonic() - started
    rss = psutil.Process().memory_info().rss
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    cameras = scene.getTrainCameras()
    if len(cameras) != args.expected_image_count:
        raise ValueError("Scene camera count differs from the declared input denominator")
    rows, payload = [], {"cpu": 0, "cuda": 0}
    for camera in cameras:
        for value in (camera.original_image, camera.alpha_mask):
            if value is not None:
                if value.device.type != args.worker:
                    raise ValueError("The native loader silently used a different device")
                payload[value.device.type] += value.numel() * value.element_size()
        rows.append({"name": camera.image_name, "rgb": fingerprint(camera.original_image),
            "alpha": fingerprint(camera.alpha_mask), "intrinsics": camera.calibrated_intrinsics,
            "view": camera.world_view_transform.cpu().tolist(),
            "projection": camera.full_proj_transform.cpu().tolist()})
    torch.cuda.synchronize()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_cuda = torch.cuda.max_memory_reserved()
    if peak_rss > 32 * 1024**3 or peak_cuda > budget or time.monotonic() - started > 300:
        raise MemoryError("Input residency exceeded the registered resource budget")
    result = {"status": "all_cameras_checked", "device": args.worker, "runtime": metadata,
        "input_version": args.input_version, "expected_image_count": args.expected_image_count,
        "source": data_identity(Path(__file__)),
        "load_seconds": load_seconds, "rss_after_load_bytes": rss, "peak_rss_bytes": peak_rss,
        "cuda_allocated_after_load_bytes": allocated, "cuda_reserved_after_load_bytes": reserved,
        "peak_cuda_reserved_bytes": peak_cuda, "gpu_budget_bytes": budget,
        "payload_bytes_by_device": payload, "trim": trim, "rows": rows}
    atomic_json(args.output / "RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["cpu", "cuda"])
    parser.add_argument("--expected-image-count", type=int, default=212)
    parser.add_argument("--input-version", default="C212_frozen")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.expected_image_count != 212 or args.input_version != "C212_frozen":
        raise ValueError("The three-trial stage campaign stays frozen; use explicit workers for new-map capacity")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise MemoryError("Insufficient free disk")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "frozen_scripts"
    snapshot.mkdir()
    names = [Path(__file__).name, "runtime.py", "full_intrinsics.py", "host_memory.py"]
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    state = {"status": "numeric", "candidate": "P26_INPUT_RESIDENCY", "counted": False,
             "sources": [data_identity(snapshot / name) for name in names], "pairs": [],
             "orders": [["cpu", "cuda"], ["cuda", "cpu"], ["cpu", "cuda"]]}
    atomic_json(args.output / "STATE.json", state)
    try:
        state["numeric_checks"] = numeric_checks()
        for index, order in enumerate(state["orders"]):
            pair = {}
            for label in order:
                output = args.output / f"trial{index}_{label}"
                command = [sys.executable, str(snapshot / Path(__file__).name), "--runtime", str(args.runtime),
                    "--dataset", str(args.dataset), "--output", str(output), "--worker", label]
                state.update(status="measuring", trial=index, variant=label, command=command)
                atomic_json(args.output / "STATE.json", state)
                with (args.output / f"trial{index}_{label}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                pair[label] = json.loads((output / "RESULT.json").read_text())
            pair["all212_values_layout_and_geometry_exact"] = pair["cpu"]["rows"] == pair["cuda"]["rows"]
            if not pair["all212_values_layout_and_geometry_exact"] or pair["cuda"]["payload_bytes_by_device"]["cpu"] != 0:
                raise ValueError("Native input equivalence or CPU release failed")
            state["pairs"].append(pair)
        rss = {label: statistics.median(p[label]["rss_after_load_bytes"] for p in state["pairs"])
               for label in ("cpu", "cuda")}
        reduction = 1 - rss["cuda"] / rss["cpu"]
        state.update(status="input_and_resource_guards_passed_training_pending" if reduction >= .2 else "rejected_memory_benefit",
            median_rss_after_load_bytes=rss, rss_reduction=reduction,
            median_load_seconds={label: statistics.median(p[label]["load_seconds"] for p in state["pairs"]) for label in rss},
            no_total_memory_or_training_speed_claim=True, training_run=False)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output / "STATE.json", state)
        raise
    atomic_json(args.output / "STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "rss_reduction", "median_rss_after_load_bytes", "median_load_seconds")}))


if __name__ == "__main__":
    main()

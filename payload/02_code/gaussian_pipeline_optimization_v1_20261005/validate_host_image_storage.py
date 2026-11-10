"""Validate exact host inputs and compare independent process memory footprints."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from full_intrinsics import camera_loader_adapter
from host_image_storage import HostImageStorage, host_image_loader
from runtime import activate_reference, atomic_json, data_identity


def numeric():
    checks = {}
    codes = torch.arange(256, dtype=torch.float32).reshape(1, 16, 16)/255
    camera = lambda name, rgb=codes.repeat(3, 1, 1).transpose(1, 2), alpha=None: SimpleNamespace(
        image_name=name, original_image=rgb, alpha_mask=alpha)
    sources = [camera("no_alpha"), camera("pixel_alpha", alpha=codes.transpose(1, 2)),
               camera("constant", alpha=torch.full((1, 16, 16), .123))]
    store = HostImageStorage()
    for source in sources:
        original = (source.original_image, source.alpha_mask)
        store.add(source)
        checks[source.image_name+"_released"] = source.original_image is None and source.alpha_mask is None
        store.activate(source)
        for label, before, after in zip(("rgb", "alpha"), original, (source.original_image, source.alpha_mask)):
            checks[source.image_name+"_"+label] = (after is None if before is None else
                before.stride() == after.stride() and before.dtype == after.dtype and
                torch.equal(before.contiguous().view(torch.int32), after.contiguous().view(torch.int32)))
        try:
            store.activate(source)
        except RuntimeError:
            checks["overlap_rejected"] = True
        store.release()
    def reject(label, source, capacity=4*1024**3):
        try:
            HostImageStorage(capacity).add(source)
        except (ValueError, MemoryError):
            checks[label] = True
        else:
            checks[label] = False
    reject("lossy_rejected", camera("lossy", rgb=torch.full((3,16,16), .12345)))
    reject("float64_rejected", camera("float64", rgb=codes.repeat(3,1,1).double()))
    reject("nonfinite_rejected", camera("nan", rgb=torch.full((3,16,16), float("nan"))))
    reject("range_rejected", camera("range", rgb=torch.full((3,16,16), 1.1)))
    reject("gradient_rejected", camera("grad", rgb=codes.repeat(3,1,1).requires_grad_()))
    reject("shape_rejected", camera("shape", alpha=torch.ones(16,16)))
    reject("capacity_rejected", camera("capacity"), 1024)
    try:
        store.add(sources[0])
    except ValueError:
        checks["duplicate_rejected"] = True
    try:
        store.activate(camera("unknown"))
    except ValueError:
        checks["unknown_rejected"] = True
    checks["inactive_float_copies_absent"] = store.receipt()["only_active_camera_has_float_inputs"]
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def fingerprint(tensor):
    if tensor is None:
        return None
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()}


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime)
    from arguments import ModelParams
    from scene import Scene, GaussianModel
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    free, _ = torch.cuda.mem_get_info()
    gpu_budget = min(12*1024**3, free-2*1024**3)
    if gpu_budget <= 0:
        raise MemoryError("Insufficient GPU headroom")
    native_parser = argparse.ArgumentParser()
    group = ModelParams(native_parser)
    native = native_parser.parse_args([])
    for key, value in json.loads((args.runtime/"MAIN_CONFIG.json").read_text())["parameters"].items():
        setattr(native, key, value)
    native.source_path = str(args.dataset.resolve())
    native.model_path = str((args.output/"scene").resolve())
    Path(native.model_path).mkdir()
    random.seed(0)
    torch.manual_seed(0)
    store = HostImageStorage() if args.worker == "compact" else None
    started = time.monotonic()
    with camera_loader_adapter(args.dataset) as manifest, host_image_loader(store) if store else nullcontext():
        scene = Scene(group.extract(native), GaussianModel(3), shuffle=False)
        torch.cuda.synchronize()
        trim_receipt = None
        if args.trim_host_after_load:
            from host_memory import trim_unused_host_pages
            trim_receipt = trim_unused_host_pages()
        load_seconds = time.monotonic()-started
        import psutil
        rss_after_load = psutil.Process().memory_info().rss
        peak_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        allocator_probe = None
        if args.allocator_probe:
            import ctypes
            import gc
            import xml.etree.ElementTree as ET
            libc = ctypes.CDLL("libc.so.6")
            libc.open_memstream.restype = ctypes.c_void_p
            libc.open_memstream.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
            libc.malloc_info.restype = ctypes.c_int
            libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
            libc.fclose.restype = ctypes.c_int
            libc.fclose.argtypes = [ctypes.c_void_p]
            libc.free.restype = None
            libc.free.argtypes = [ctypes.c_void_p]
            libc.malloc_trim.restype = ctypes.c_int
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            def snapshot(label):
                buffer, size = ctypes.c_void_p(), ctypes.c_size_t()
                stream = libc.open_memstream(ctypes.byref(buffer), ctypes.byref(size))
                if not stream:
                    raise OSError("Cannot open the bounded allocator diagnostic stream")
                try:
                    result = libc.malloc_info(0, stream)
                finally:
                    libc.fclose(stream)
                try:
                    if result != 0 or size.value > 1024*1024:
                        raise ValueError("Invalid allocator XML response")
                    xml = ctypes.string_at(buffer.value, size.value).decode("ascii")
                finally:
                    libc.free(buffer)
                (args.output/(label+".xml")).write_text(xml)
                root = ET.fromstring(xml)
                return {"rss_bytes": psutil.Process().memory_info().rss, "heap_count": len(root.findall("heap")),
                        "allocator_xml": label+".xml",
                        "totals": {row.attrib["type"]: int(row.attrib["size"]) for row in root.findall("total")},
                        "system": {row.attrib["type"]: int(row.attrib["size"]) for row in root.findall("system")}}
            allocator_probe = {"api": "malloc_info_XML", "before_gc": snapshot("before_gc")}
            allocator_probe["collected_objects"] = gc.collect()
            allocator_probe["after_gc"] = snapshot("after_gc")
            probe_started = time.monotonic()
            allocator_probe["malloc_trim_returned"] = libc.malloc_trim(0)
            allocator_probe["trim_seconds"] = time.monotonic()-probe_started
            allocator_probe["after_trim"] = snapshot("after_trim")
            atomic_json(args.output/"ALLOCATOR_PROBE.json", allocator_probe)
        cameras = scene.getTrainCameras()
        if len(cameras) != 212:
            raise ValueError("The frozen camera denominator changed")
        rows = []
        for camera in cameras:
            if store:
                store.activate(camera)
            rows.append({"name": camera.image_name, "rgb": fingerprint(camera.original_image),
                "alpha": fingerprint(camera.alpha_mask), "intrinsics": camera.calibrated_intrinsics,
                "view": camera.world_view_transform.cpu().tolist(), "projection": camera.full_proj_transform.cpu().tolist()})
            if store:
                store.release()
    receipt = store.receipt() if store else None
    checks = {"memory_budget": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 32*1024**3,
        "GPU_budget": torch.cuda.max_memory_reserved() <= gpu_budget,
        "wall_budget": time.monotonic()-started <= 300,
        "all212_camera_geometry_bound": len(manifest) == len(rows) == 212,
        "no_inactive_float_copies": receipt["only_active_camera_has_float_inputs"] if store else True}
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "variant": args.worker, "source": data_identity(Path(__file__)),
        "adapter": data_identity(Path(__file__).with_name("host_image_storage.py")), "runtime": metadata,
        "load_seconds": load_seconds, "rss_after_load_bytes": rss_after_load,
        "allocator_probe": allocator_probe,
        "trim_host_after_load": trim_receipt,
        "peak_rss_after_load_bytes": peak_after_load, "storage": receipt, "rows": rows,
        "elapsed_seconds": time.monotonic()-started}
    atomic_json(args.output/"RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("The independent memory diagnostic failed its guard")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--worker", choices=["native", "compact"])
    parser.add_argument("--allocator-probe", action="store_true")
    parser.add_argument("--trim-host-after-load", action="store_true")
    args = parser.parse_args()
    if args.allocator_probe and not args.worker:
        raise ValueError("Allocator attribution is a separate worker diagnostic, not an acceptance retry")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for name in ("validate_host_image_storage.py", "host_image_storage.py", "host_memory.py", "runtime.py", "full_intrinsics.py"):
        shutil.copyfile(Path(__file__).with_name(name), snapshot/name)
    state = {"status": "numeric", "orders": [["native", "compact"], ["compact", "native"], ["native", "compact"]],
             "source": data_identity(Path(__file__)), "adapter": data_identity(Path(__file__).with_name("host_image_storage.py")),
             "counted": False}
    state["equal_trim_in_both_variants"] = args.trim_host_after_load
    atomic_json(args.output/"STATE.json", state)
    state["numeric_checks"] = numeric()
    pairs = []
    try:
        for trial, order in enumerate(state["orders"]):
            pair = {}
            for variant in order:
                output = args.output/f"trial{trial}_{variant}"
                command = [sys.executable, str(Path(__file__)), "--runtime", str(args.runtime),
                    "--dataset", str(args.dataset), "--output", str(output), "--worker", variant]
                if args.trim_host_after_load:
                    command.append("--trim-host-after-load")
                state.update(status="measuring", trial=trial, variant=variant, command=command)
                atomic_json(args.output/"STATE.json", state)
                with (args.output/f"trial{trial}_{variant}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                pair[variant] = json.loads((output/"RESULT.json").read_text())
            pair["all212_inputs_and_geometry_exact"] = pair["native"]["rows"] == pair["compact"]["rows"]
            if not pair["all212_inputs_and_geometry_exact"]:
                raise AssertionError("Actual image values, strides or camera geometry changed")
            pairs.append(pair)
        rss = {variant: float(np.median([pair[variant]["rss_after_load_bytes"] for pair in pairs])) for variant in ("native", "compact")}
        reduction = 1-rss["compact"]/rss["native"]
        passed = reduction >= .2 and all(pair["compact"]["storage"]["payload_reduction"] >= .7 for pair in pairs)
        state.update(status="input_and_memory_guards_passed_training_pending" if passed else "rejected_memory_benefit",
            median_rss_after_load_bytes=rss, rss_reduction=reduction,
            median_load_seconds={variant: float(np.median([pair[variant]["load_seconds"] for pair in pairs])) for variant in ("native", "compact")},
            pairs=pairs, training_run=False)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "median_rss_after_load_bytes", "rss_reduction", "median_load_seconds")}), flush=True)


if __name__ == "__main__":
    main()

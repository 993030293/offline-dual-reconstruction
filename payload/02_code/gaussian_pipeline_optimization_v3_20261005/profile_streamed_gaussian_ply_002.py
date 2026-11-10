"""Verify complete native PLY equivalence and measure isolated fresh save workers."""

import argparse
import gc
import json
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from streamed_gaussian_ply_oracle_002 import NAMES, admission, streamed_ply_model, verify_complete_bytes


def file_exact(left, right):
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            x, y = a.read(1024**2), b.read(1024**2)
            if x != y:
                return False
            if not x:
                return True


def state(model, torch):
    return {name: (getattr(model, name).detach().clone(), getattr(model, name).grad,
        tuple(getattr(model, name).stride()), getattr(model, name).storage_offset(),
        getattr(model, name).requires_grad, getattr(model, name)._version) for name in NAMES}


def state_exact(model, snapshot, torch):
    return all(torch.equal(getattr(model, name).detach().view(torch.int32), value[0].view(torch.int32))
        and getattr(model, name).grad is value[1] and tuple(getattr(model, name).stride()) == value[2]
        and getattr(model, name).storage_offset() == value[3]
        and getattr(model, name).requires_grad == value[4] and getattr(model, name)._version == value[5]
        for name, value in snapshot.items())


def rng(torch):
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()]


def rng_exact(left, right, torch):
    return (left[0] == right[0] and left[1][0] == right[1][0]
        and np.array_equal(left[1][1], right[1][1]) and left[1][2:] == right[1][2:]
        and torch.equal(left[2], right[2]) and len(left[3]) == len(right[3])
        and all(torch.equal(a, b) for a, b in zip(left[3], right[3])))


def generic(output, GaussianModel, torch):
    from ply_serialization import vectorized_ply_model
    plain, candidate = vectorized_ply_model(GaussianModel, True), streamed_ply_model(GaussianModel)
    rows = []
    torch.manual_seed(127)
    for degree in (0, 1, 3):
        for count in (0, 1, 7, 257):
            shapes = ((count, 3), (count, 1, 3), (count, (degree + 1)**2 - 1, 3),
                      (count, 1), (count, 3), (count, 4))
            values = [torch.randn(shape, device="cuda") for shape in shapes]
            if count:
                values[0][0, :] = torch.tensor([-0., 1.401298464324817e-45, 3.4028234663852886e38], device="cuda")
            # A noncontiguous layout preserves the exact original transpose and flatten.
            if count > 1:
                values[2] = values[2].transpose(0, 1).contiguous().transpose(0, 1)
            models, paths = [], []
            for mode, cls in (("native", plain), ("stream", candidate)):
                model = cls(degree)
                for name, value in zip(NAMES, values):
                    setattr(model, name, torch.nn.Parameter(value.clone(memory_format=torch.preserve_format)))
                snapshot, before = state(model, torch), rng(torch)
                folder = output / f"SH{degree}_N{count}" / mode
                path = folder / "point_cloud.ply"
                model.save_ply(str(path))
                checks = {"readonly_parameters_grad_layout_versions": state_exact(model, snapshot, torch),
                    "all_RNG_states_unchanged": rng_exact(before, rng(torch), torch),
                    "point_and_oracle_complete_bytes": file_exact(path, folder / "native_oracle.ply")}
                rows.append({"degree": degree, "count": count, "mode": mode, "checks": checks})
                paths.append(path)
                models.append(model)
            rows.append({"degree": degree, "count": count, "checks": {"unmodified_reference_and_stream_all_bytes": file_exact(*paths)}})
    model = models[-1]
    clean = paths[-1].read_bytes()
    header_end = clean.index(b"end_header\n") + len(b"end_header\n")
    mutations = {"header_type": clean.replace(b"property float x\n", b"property int x\n", 1),
        "truncate_one_byte": clean[:-1], "extra_EOF": clean + b"x",
        "XYZ_one_bit": clean[:header_end] + bytes([clean[header_end] ^ 1]) + clean[header_end + 1:],
        "last_rotation_bit": clean[:-1] + bytes([clean[-1] ^ 1]),
        "SH_bit": clean[:header_end + 40] + bytes([clean[header_end + 40] ^ 1]) + clean[header_end + 41:]}
    row_bytes = len(model.construct_list_of_attributes()) * 4
    mutations["row_order"] = clean[:header_end] + clean[header_end + row_bytes:header_end + 2 * row_bytes] + clean[header_end:header_end + row_bytes] + clean[header_end + 2 * row_bytes:]
    for name, payload in mutations.items():
        path = output / (name + ".ply")
        path.write_bytes(payload)
        rejected = False
        try:
            verify_complete_bytes(model, path)
        except AssertionError as error:
            rejected = str(error) == "Vectorized serialization differs from the native file bytes"
        rows.append({"mutation": name, "checks": {"full_byte_mismatch_rejected": rejected}})
    existing = output / "existing_oracle"
    existing.mkdir()
    (existing / "native_oracle.ply").write_bytes(b"protected")
    count_before = len(model.ply_serialization_events)
    rejected = False
    try:
        model.save_ply(str(existing / "point_cloud.ply"))
    except FileExistsError as error:
        rejected = str(error) == "Never overwrite an existing serialization oracle"
    rows.append({"checks": {"original_existing_oracle_error_and_prefix": rejected and len(model.ply_serialization_events) == count_before
        and (existing / "point_cloud.ply").is_file() and (existing / "native_oracle.ply").read_bytes() == b"protected"}})
    for kind in ("nan", "inf", "float64", "cpu"):
        models = [plain(3), candidate(3)]
        errors, paths = [], []
        for index, current in enumerate(models):
            for name in NAMES:
                value = getattr(model, name).detach().clone()
                if kind == "float64": value = value.double()
                if kind == "cpu": value = value.cpu()
                if kind in ("nan", "inf") and name == "_xyz": value[0, 0] = float(kind)
                setattr(current, name, torch.nn.Parameter(value))
            path = output / (kind + str(index)) / "point_cloud.ply"
            try:
                current.save_ply(str(path))
                errors.append(None)
            except (ValueError, AssertionError) as error:
                errors.append((type(error).__name__, str(error)))
            paths.append(path)
        rows.append({"unsupported": kind, "checks": {"original_selected_before_execution": not admission(models[1], GaussianModel)[0],
            "original_error_type_and_message": errors[0] == errors[1],
            "original_output_presence_and_bytes": paths[0].exists() == paths[1].exists() and
                (not paths[0].exists() or file_exact(*paths))}})
    checks = {"all_generic_full_bytes_and_original_guards": all(all(row["checks"].values()) for row in rows)}
    atomic_json(output / "RESULT.json", {"status": "generic_passed" if all(checks.values()) else "generic_failed",
        "checks": checks, "rows": rows, "counted": False})
    if not all(checks.values()): raise AssertionError("A generic original PLY contract failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generic", "native", "stream"], required=True)
    for name in ("runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    args.output.resolve().relative_to((ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dpv001").resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    reference = activate_reference(args.runtime)
    import torch
    import plyfile
    from scene.gaussian_model import GaussianModel
    from ply_serialization import vectorized_ply_model
    from vectorized_gaussian_read import vectorized_gaussian_read
    torch.set_num_threads(4)
    paths = [Path(__file__), Path(__file__).with_name("streamed_gaussian_ply_oracle_002.py"),
        Path(sys.modules[GaussianModel.__module__].__file__), Path(plyfile.__file__),
        V1 / "ply_serialization.py", V1 / "runtime.py", V2 / "vectorized_gaussian_read.py"]
    if args.model: paths.append(args.model)
    identities = [data_identity(path) for path in paths]
    atomic_json(args.output / "CONTRACT.json", {"sources_inputs": identities, "reference": reference,
        "mode": args.mode, "all_original_native_byte_oracle_artifacts_kept": True, "counted": False})
    if args.mode == "generic":
        generic(args.output, GaussianModel, torch)
        if identities != [data_identity(path) for path in paths]: raise ValueError("Generic sources changed")
        return
    if args.model is None: raise ValueError("The original complete mature model is required")
    cls = vectorized_ply_model(GaussianModel, True) if args.mode == "native" else streamed_ply_model(GaussianModel)
    model = cls(3)
    with vectorized_gaussian_read(True): model.load_ply(str(args.model))
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    budget = min(12 * 1024**3, free - 2 * 1024**3)
    torch.cuda.reset_peak_memory_stats()
    before = time.perf_counter()
    if identities != [data_identity(path) for path in paths]: raise ValueError("Source/input changed before actual save")
    snapshot, previous_rng = state(model, torch), rng(torch)
    point = args.output / "point_cloud.ply"
    model.save_ply(str(point))
    checks = {"actual_saved_point_count": len(model._xyz) == 526951,
        "complete_original_parameters_grad_layout_versions": state_exact(model, snapshot, torch),
        "all_RNG_unchanged": rng_exact(previous_rng, rng(torch), torch),
        "full_native_point_and_reference_bytes": file_exact(point, args.output / "native_oracle.ply"),
        "complete_source_and_input_unchanged": identities == [data_identity(path) for path in paths]}
    roundtrip = GaussianModel(3)
    with vectorized_gaussian_read(True): roundtrip.load_ply(str(point))
    checks["unmodified_loader_all_six_parameter_bits"] = all(torch.equal(getattr(model, name).view(torch.int32),
        getattr(roundtrip, name).view(torch.int32)) for name in NAMES)
    checks["CUDA_budget"] = torch.cuda.max_memory_reserved() <= budget
    events = list(model.ply_serialization_events)
    payload = data_identity(point)
    del model, snapshot, roundtrip
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - before
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    checks["WSL_RSS_budget"] = rss <= 32 * 1024**3
    result = {"status": "complete_native_save_verified" if all(checks.values()) else "required_save_guard_failed",
        "mode": args.mode, "checks": checks, "complete_phase_seconds": elapsed, "events": events,
        "complete_child_seconds": time.perf_counter() - STARTED, "WSL_peak_RSS_bytes": rss,
        "CUDA_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "point_payload": payload, "counted": False}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "mode", "checks", "complete_phase_seconds", "complete_child_seconds", "WSL_peak_RSS_bytes")}), flush=True)
    if not all(checks.values()): raise AssertionError("A complete original PLY save guard failed")


if __name__ == "__main__":
    main()

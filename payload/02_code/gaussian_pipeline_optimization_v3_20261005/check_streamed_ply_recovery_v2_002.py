"""Check real permission failures, cancellation prefixes, and fresh recovery."""

import builtins
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]

import psutil

from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


@contextmanager
def cancel_write(target, kind):
    original = builtins.open
    flags = {name: getattr(shutil, name) for name in ("_USE_CP_SENDFILE", "_USE_CP_COPY_FILE_RANGE") if hasattr(shutil, name)}
    calls = []
    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, payload):
            if not payload:
                return self.stream.write(payload)
            calls.append(len(payload))
            self.stream.write(payload[:10])
            self.stream.flush()
            raise kind("owned_PLY_cancel")
    def opened(file, mode="r", *args, **kwargs):
        stream = original(file, mode, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file) == target and "w" in mode:
            return PartialWriter(stream)
        return stream
    try:
        for name in flags:
            setattr(shutil, name, False)
        builtins.open = opened
        yield calls
    finally:
        builtins.open = original
        for name, value in flags.items():
            setattr(shutil, name, value)


def main():
    output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dpv001/v2r1"
    output.mkdir(exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid,
        "creation_time": process.create_time()})
    with LinuxRSSGuard(output, atomic_json, maximum_seconds=60):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from scene.gaussian_model import GaussianModel
        from ply_serialization import vectorized_ply_model
        import streamed_gaussian_ply_oracle_002 as implementation
        from profile_streamed_gaussian_ply_002 import state, state_exact, rng, rng_exact, file_exact
        torch.set_num_threads(4)
        torch.manual_seed(127)
        free, total = torch.cuda.mem_get_info()
        if total - free > 12 * 1024**3 or free < 2 * 1024**3:
            raise MemoryError("Original CUDA headroom is unavailable")
        sources = [data_identity(path) for path in (Path(__file__), Path(implementation.__file__),
            Path(sys.modules[GaussianModel.__module__].__file__), CODE / "mapped_image_linux_guard.py",
            ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/ply_serialization.py")]
        classes = {"native": vectorized_ply_model(GaussianModel, True),
            "stream": implementation.streamed_ply_model(GaussianModel)}
        values = [torch.randn(shape, device="cuda") for shape in ((7, 3), (7, 1, 3),
            (7, 15, 3), (7, 1), (7, 3), (7, 4))]
        rows, temporary = [], []
        original_uid = os.geteuid()
        cases = [("cancel", target, kind) for target in ("point_cloud.ply", "native_oracle.ply")
            for kind in (KeyboardInterrupt, SystemExit)]
        cases += [("permission", target, PermissionError) for target in ("point_cloud.ply", "native_oracle.ply")]
        for previous in (False, True):
            for case, target_name, kind in cases:
                label = f"previous{int(previous)}_{case}_{target_name}_{kind.__name__}"
                states, recovered = {}, []
                for mode, cls in classes.items():
                    model = cls(3)
                    for name, value in zip(implementation.NAMES, values):
                        setattr(model, name, torch.nn.Parameter(value.clone()))
                    base = output / label / mode
                    base.mkdir(parents=True)
                    if previous:
                        model.save_ply(str(base / "previous/point_cloud.ply"))
                    prefix = list(getattr(model, "ply_serialization_events", []))
                    attribute = hasattr(model, "ply_serialization_events")
                    snapshot, before_rng = state(model, torch), rng(torch)
                    if not implementation.admission(model, GaussianModel)[0]:
                        raise AssertionError("The actual candidate must remain on its admitted route")
                    error, calls = None, []
                    folder = base / "failed"
                    if case == "permission":
                        folder = Path(tempfile.mkdtemp(prefix="r127_v2_permission_"))
                        temporary.append(str(folder))
                        if target_name == "native_oracle.ply":
                            point = folder / "point_cloud.ply"
                            point.write_bytes(b"owned_replaceable_point")
                            point.chmod(0o666)
                        folder.chmod(0o555)
                        try:
                            if original_uid == 0:
                                os.seteuid(65534)
                            model.save_ply(str(folder / "point_cloud.ply"))
                        except PermissionError as caught:
                            error = (type(caught).__name__, caught.errno, str(caught).replace(str(folder), "<case>"))
                        finally:
                            if os.geteuid() != original_uid:
                                os.seteuid(original_uid)
                            folder.chmod(0o700)
                    else:
                        target = folder / target_name
                        try:
                            with cancel_write(target, kind) as calls:
                                model.save_ply(str(folder / "point_cloud.ply"))
                        except (KeyboardInterrupt, SystemExit) as caught:
                            error = (type(caught).__name__, None, str(caught))
                    content = {}
                    for name in ("point_cloud.ply", "native_oracle.ply"):
                        file = folder / name
                        content[name] = None if not file.exists() else file.read_bytes().hex()
                    states[mode] = {"error": error, "failure_target_received_one_cancel":
                        len(calls) == 1 if case == "cancel" else error is not None and error[1] == 13,
                        "failed_file_prefixes": content, "event_attribute_same": hasattr(model, "ply_serialization_events") == attribute,
                        "original_event_prefix_kept": getattr(model, "ply_serialization_events", []) == prefix,
                        "parameters_grad_layout_versions_unchanged": state_exact(model, snapshot, torch),
                        "all_RNG_unchanged": rng_exact(before_rng, rng(torch), torch),
                        "effective_uid_restored": os.geteuid() == original_uid}
                    if case == "permission":
                        retained = base / "failed_permission_evidence"
                        retained.mkdir()
                        for name in content:
                            if content[name] is not None:
                                shutil.copyfile(folder / name, retained / name)
                    recovery = base / "fresh_recovery/point_cloud.ply"
                    model.save_ply(str(recovery))
                    recovered.append(recovery)
                    states[mode]["fresh_recovery_original_event_count"] = len(model.ply_serialization_events) == len(prefix) + 1
                    states[mode]["fresh_recovery_all_native_reference_bytes"] = file_exact(recovery, recovery.with_name("native_oracle.ply"))
                    states[mode]["recovery_parameters_and_RNG_unchanged"] = (
                        state_exact(model, snapshot, torch) and rng_exact(before_rng, rng(torch), torch))
                    if mode == "stream":
                        states[mode]["fresh_recovery_actual_stream_route"] = "streamed_native_verification" in model.ply_serialization_events[-1]
                comparable = [{key: value for key, value in states[mode].items()
                    if key != "fresh_recovery_actual_stream_route"} for mode in ("native", "stream")]
                checks = {"original_exceptions_prefixes_events_and_state_exact": comparable[0] == comparable[1],
                    "all_state_recovery_and_error_checks": all(value for state_value in comparable
                        for key, value in state_value.items() if isinstance(value, bool)),
                    "both_expected_failure_types": all(state_value["error"] and state_value["error"][0] == kind.__name__ for state_value in comparable),
                    "all_recovered_original_point_bytes_exact": file_exact(*recovered),
                    "actual_stream_used_on_recovery": states["stream"]["fresh_recovery_actual_stream_route"]}
                rows.append({"case": label, "states": states, "checks": checks})
        checks = {"all12_permission_cancel_and_fresh_recovery_pairs": len(rows) == 12
            and all(all(row["checks"].values()) for row in rows),
            "effective_uid_restored": os.geteuid() == original_uid,
            "source_inputs_unchanged": all(data_identity(item["path"]) == item for item in sources),
            "bounded_owned_temporary_files": len(temporary) == 8
                and sum(sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file()) for folder in temporary) < 1024**2}
        result = {"status": "original_permission_cancel_and_recovery_verified" if all(checks.values()) else "failed_verification",
            "checks": checks, "rows": rows, "sources": sources, "owned_temporary_permission_folders": temporary,
            "complete_child_seconds_before_guard_exit": time.perf_counter() - STARTED,
            "new_optimization_count": 0, "original_uid": original_uid, "effective_uid_after": os.geteuid()}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "complete_child_seconds_before_guard_exit")}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A real original permission, cancellation or recovery contract differs")


if __name__ == "__main__":
    main()


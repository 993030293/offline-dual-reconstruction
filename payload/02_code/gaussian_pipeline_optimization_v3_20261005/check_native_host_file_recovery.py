"""Exercise original fallback prefixes and real host cancellation with fresh recovery."""

import time

STARTED = time.perf_counter()

import gc
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import threading

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from native_host_gaussian_files import host_ply_model
import check_streamed_ply_recovery_v2_002 as prior


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("The original permission and fallback contract source changed")
    return source.replace(old, new, 1)


def main():
    output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/r0"
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid,
        "creation_time": process.create_time(), "counted": False})
    paths = [Path(__file__), Path(prior.__file__), CODE / "native_host_gaussian_files.py",
        CODE / "native_host_file_worker.py", CODE / "streamed_gaussian_ply_oracle_002.py"]
    identities = [data_identity(path) for path in paths]
    with LinuxRSSGuard(output, atomic_json, maximum_seconds=120):
        events, cancel_folders = [], []
        source = inspect.getsource(prior.main)
        source = replace_once(source, 'output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dpv001/v2r1"',
            'output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/r0/fallback"')
        source = replace_once(source, '"stream": implementation.streamed_ply_model(GaussianModel)',
            '"stream": owned_factory(GaussianModel)')
        source = replace_once(source, 'folder = base / "failed"',
            'folder = base / "failed"\n                    if case == "cancel":\n                        folder = Path(tempfile.mkdtemp(prefix="r133_fallback_cancel_"))\n                        owned_cancel_folders.append(str(folder))')
        source = replace_once(source, 'if case == "permission":\n                        retained = base / "failed_permission_evidence"',
            'if case in ("permission", "cancel"):\n                        retained = base / "failed_permission_evidence"')
        namespace = dict(prior.__dict__)
        namespace.update(owned_factory=lambda base: host_ply_model(base, events),
            owned_cancel_folders=cancel_folders)
        exec(compile(source, "<original_fallback_permission_cancel_with_real_host_recovery>", "exec"), namespace)
        namespace["main"]()
        fallback = json.loads((output / "fallback/RESULT.json").read_text(encoding="utf-8"))
        import torch
        from scene.gaussian_model import GaussianModel
        from ply_serialization import vectorized_ply_model
        from profile_streamed_gaussian_ply_002 import file_exact, state, state_exact, rng, rng_exact
        from streamed_gaussian_ply_oracle_002 import NAMES
        torch.set_num_threads(4)
        count = 70000
        shapes = ((count, 3), (count, 1, 3), (count, 15, 3), (count, 1), (count, 3), (count, 4))
        values = [torch.randn(shape, device="cuda") for shape in shapes]
        rows = []
        for previous in (False, True):
            for kind, sig in ((KeyboardInterrupt, signal.SIGINT), (SystemExit, signal.SIGUSR1)):
                folder = output / f"host_previous{int(previous)}_{kind.__name__}"
                model = host_ply_model(GaussianModel, events)(3)
                for name, value in zip(NAMES, values):
                    setattr(model, name, torch.nn.Parameter(value.clone()))
                if previous:
                    model.save_ply(str(folder / "previous/point_cloud.ply"))
                prefix = list(getattr(model, "ply_serialization_events", []))
                attribute = hasattr(model, "ply_serialization_events")
                snapshot, before_rng = state(model, torch), rng(torch)
                target = folder / "interrupted/point_cloud.ply"
                receipt = target.parent / "HOST_IO_RESOURCE.json"
                stop = threading.Event()
                observed = {}
                old_handler = signal.getsignal(sig)

                def interrupted(signum, frame):
                    raise kind("owned_host_cancel")

                def observe_then_interrupt():
                    try:
                        while not stop.wait(.001):
                            if receipt.is_file():
                                row = json.loads(receipt.read_text(encoding="utf-8"))
                                if row["status"] == "running":
                                    observed.update(row)
                                    os.kill(os.getpid(), sig)
                                    return
                            if time.perf_counter() - monitor_started > 10:
                                observed["failed_to_observe_running_host"] = True
                                return
                    except BaseException as error:
                        observed["observer_error"] = repr(error)

                monitor_started = time.perf_counter()
                observer = threading.Thread(target=observe_then_interrupt, name="owned-host-cancel-probe", daemon=True)
                signal.signal(sig, interrupted)
                observer.start()
                caught = None
                try:
                    model.save_ply(str(target))
                except (KeyboardInterrupt, SystemExit) as error:
                    caught = {"type": type(error).__name__, "message": str(error)}
                finally:
                    stop.set()
                    observer.join(timeout=12)
                    signal.signal(sig, old_handler)
                row = json.loads(receipt.read_text(encoding="utf-8"))
                event_prefix_after_cancel = list(getattr(model, "ply_serialization_events", []))
                event_attribute_after_cancel = hasattr(model, "ply_serialization_events")
                retry_error = None
                try:
                    model.save_ply(str(target))
                except FileExistsError as error:
                    retry_error = str(error)
                recovery = folder / "fresh_recovery/point_cloud.ply"
                model.save_ply(str(recovery))
                original = vectorized_ply_model(GaussianModel, True)(3)
                for name, value in zip(NAMES, values):
                    setattr(original, name, torch.nn.Parameter(value.clone()))
                reference = folder / "unmodified_reference/point_cloud.ply"
                original.save_ply(str(reference))
                checks = {
                    "actual_running_host_observed_before_signal": observed.get("status") == "running",
                    "expected_caller_cancellation_type_message": caught == {"type": kind.__name__, "message": "owned_host_cancel"},
                    "owned_observer_joined": not observer.is_alive(),
                    "real_host_guard_finished_before_propagation": row["status"] == "completed",
                    "original_event_attribute_and_prefix_kept": event_attribute_after_cancel == attribute
                        and event_prefix_after_cancel == prefix and model.ply_serialization_events[:-1] == prefix,
                    "interrupted_complete_copy_is_preserved_without_event": file_exact(target, target.with_name("native_oracle.ply")),
                    "original_same_directory_retry_rejected": retry_error == "Never overwrite an existing serialization oracle",
                    "fresh_recovery_complete_native_files_and_event": file_exact(recovery, reference)
                        and file_exact(recovery, recovery.with_name("native_oracle.ply"))
                        and len(model.ply_serialization_events) == len(prefix) + 1,
                    "all_parameter_bits_grad_layout_versions_readonly": state_exact(model, snapshot, torch),
                    "all_RNG_unchanged": rng_exact(before_rng, rng(torch), torch),
                    "signal_handler_restored": signal.getsignal(sig) == old_handler}
                rows.append({"previous_event": previous, "cancellation": kind.__name__, "checks": checks,
                    "actual_host_resource": row, "observed_running_host": observed,
                    "prior_event_attribute": attribute, "error": caught,
                    "partial_file_policy": "preserve_physically_complete_drained_copy_without_committing_an_event"})
                del model, snapshot, original
                gc.collect()
                torch.cuda.empty_cache()
                atomic_json(output / "PARTIAL.json", {"rows": rows, "counted": False})
        checks = {
            "all12_original_permission_cancel_fallback_and_real_recovery_pairs": all(fallback["checks"].values()),
            "all_four_real_host_cancellation_and_fresh_recovery_rows": len(rows) == 4 and all(all(row["checks"].values()) for row in rows),
            "all_source_inputs_unchanged": identities == [data_identity(path) for path in paths]}
        result = {"status": "host_cancel_fallback_permission_and_recovery_verified" if all(checks.values()) else "failed_verification",
            "checks": checks, "rows": rows, "host_events": events, "sources": identities,
            "owned_cancel_folders": cancel_folders, "original_fallback_result": fallback,
            "complete_child_seconds": time.perf_counter() - STARTED, "counted": False}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "complete_child_seconds")}), flush=True)
        if not all(checks.values()):
            raise AssertionError("An actual host cancellation or original fallback contract failed")


if __name__ == "__main__":
    main()

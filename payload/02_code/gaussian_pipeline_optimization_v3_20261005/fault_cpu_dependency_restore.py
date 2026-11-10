"""Inject an owned checksum error or actual SIGINT into the full native entry."""

import argparse
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fault", choices=("checksum_error", "actual_cancel"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    bridge = CODE / "profile_cpu_dependency_restore.py"
    tree = ast.parse(bridge.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignment = next(node for node in function.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "replacements" for target in node.targets))
    replacements = ast.literal_eval(assignment.value)
    source = (CODE / "profile_native_restore_cost.py").read_text(encoding="utf-8")
    for before, after in replacements:
        if source.count(before) != 1:
            raise AssertionError("An original source-bound complete entry changed")
        source = source.replace(before, after)
    before = "    source_ids = [data_identity(path) for path in sources]"
    after = "    sources.append(CODE / 'profile_cpu_dependency_restore.py')\n" + before
    if source.count(before) != 1:
        raise AssertionError("The original full source boundary changed")
    source = source.replace(before, after)
    marker = "    output.mkdir(parents=True, exist_ok=False)"
    source = source.replace(marker, marker + "\n    (output / 'EXECUTED_SOURCE.py').write_text(_executed_source, encoding='utf-8')", 1)
    if args.fault == "actual_cancel":
        before = "            identities = [data_identity(path) for path in inputs]"
        after = ("            import signal, os, threading\n"
            "            live = [{'name': thread.name, 'ident': thread.ident} for thread in threading.enumerate() if thread.name.startswith('owned-CPU-dependency') and thread.is_alive()]\n"
            "            if not live: raise AssertionError('The actual cancellation has no live owned import worker')\n"
            "            atomic_json(output / 'ACTUAL_SIGNAL.json', {'signal': int(signal.SIGINT), 'pid': os.getpid(), 'live_owned_workers': live})\n"
            "            os.kill(os.getpid(), signal.SIGINT)\n" + before)
    else:
        before = "                iteration = restore.prepare(scene.gaussians, opt_group.extract(arguments), scene)"
        after = ("                owned.resolve().relative_to(output.resolve())\n"
            "                with owned.open('r+b') as stream:\n"
            "                    byte = stream.read(1)\n"
            "                    stream.seek(0)\n"
            "                    stream.write(bytes([byte[0] ^ 1]))\n"
            "                    stream.flush()\n"
            "                atomic_json(output / 'OWNED_CHECKSUM_FAULT.json', {'only_owned_checkpoint_changed': str(owned), 'original': data_identity(CHECKPOINT)})\n" + before)
    if source.count(before) != 1:
        raise AssertionError("An original actual fault boundary changed")
    source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_actual_CPU_dependency_fault",
                 "_executed_source": source}
    previous = sys.argv
    sys.argv = [str(bridge), "--output", str(args.output), "--mode", "bootstrap"]
    expected = False
    binding_records = []
    try:
        exec(compile(source, "<owned_actual_CPU_dependency_fault>", "exec"), namespace)
        original_context = namespace["timed_recovery"]

        @contextmanager
        def observed_context(adapter, gaussians, torch, device_layout):
            before = {name: getattr(adapter, name) for name in ("exact_equal", "data_identity", "atomic_json")}
            loader = torch.load
            row = {"actual_context_entered": True}
            binding_records.append(row)
            try:
                with original_context(adapter, gaussians, torch, device_layout) as trace:
                    row["actual_trace"] = trace
                    yield trace
            finally:
                row["actual_original_binding_identities_restored"] = all(getattr(adapter, name) is value for name, value in before.items())
                row["actual_previous_private_loader_identity_restored"] = torch.load is loader

        namespace["timed_recovery"] = observed_context
        namespace["main"]()
    except BaseException as error:
        expected = (isinstance(error, KeyboardInterrupt) if args.fault == "actual_cancel" else
                    isinstance(error, ValueError) and str(error) == "Checkpoint payload identity mismatch")
        torch = sys.modules.get("torch")
        adapter = sys.modules.get("checkpoint_adapter")
        checks = {
            "actual_expected_original_exception": expected,
            "all_owned_CPU_import_workers_joined": not any(thread.name.startswith("owned-CPU-dependency") for thread in threading.enumerate()),
            "original_recovery_bindings_restored": (args.fault == "actual_cancel" and adapter is None and not binding_records)
                or len(binding_records) == 1 and all(row["actual_original_binding_identities_restored"]
                    and row["actual_previous_private_loader_identity_restored"] for row in binding_records),
            "original_native_Torch_loader_restored": torch is not None and torch.load is torch.serialization.load,
            "cancel_has_no_CUDA_context": args.fault != "actual_cancel" or not torch.cuda.is_initialized(),
        }
        atomic_json(args.output / "FAULT_RESULT.json", {"status": "actual_expected_fault_propagated" if all(checks.values()) else "failed_fault_verification",
            "fault": args.fault, "checks": checks, "error": {"type": type(error).__name__, "message": str(error)},
            "process": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()}, "actual_binding_records": binding_records,
            "actual_seconds_before_publication": time.perf_counter() - started, "script": data_identity(Path(__file__)), "new_optimization_count": 0})
        raise
    finally:
        sys.argv = previous
    if not expected:
        raise AssertionError("The actual fault did not propagate")


if __name__ == "__main__":
    main()

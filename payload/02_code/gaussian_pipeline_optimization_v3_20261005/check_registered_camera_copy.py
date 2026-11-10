"""Exercise real CUDA registration, exact copying, ownership and cleanup failures."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from registered_camera_copy import HostRegistration, RegisteredCameraCopy


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.resolve().relative_to((ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/registered_camera_copy_001").resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    import psutil
    import torch
    torch.set_num_threads(4)
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time(), "scope": "WSL"})
    checks = {}

    def refuses(name, callback, kind=ValueError):
        try:
            callback()
        except kind:
            checks[name] = True
        else:
            raise AssertionError(name + " did not reject the invalid operation")

    sources = [torch.arange(3*17*19, dtype=torch.float32).reshape(3,17,19),
               torch.arange(3*17*19, dtype=torch.float32).reshape(17,19,3).permute(2,0,1)]
    for index, source in enumerate(sources):
        original = source.clone(memory_format=torch.preserve_format)
        transfer = RegisteredCameraCopy("registered", [source, source])
        try:
            for _ in range(3):
                transfer.begin(source)
                copied = transfer.consume()
                checks["exact_layout_values_"+str(index)] = (copied.dtype == source.dtype
                    and copied.stride() == source.stride() and torch.equal(copied.cpu(), source))
            refuses("foreign_source_"+str(index), lambda: transfer.begin(source.clone()))
        finally:
            transfer.close()
        checks["owned_single_registration_closed_"+str(index)] = (not source.is_pinned()
            and transfer.trace["registration"]["registrations"] == 1
            and transfer.trace["registration"]["unregistrations"] == 1
            and transfer.trace["duplicate_CPU_staging_bytes"] == 0
            and torch.equal(source, original) and transfer.closed)
        transfer.close()
        refuses("closed_consumer_"+str(index), transfer.consume)
    refuses("invalid_dtype", lambda: HostRegistration([sources[0].double()]))
    refuses("empty_bank", lambda: HostRegistration([]))
    refuses("zero_host_budget", lambda: HostRegistration([sources[0]], maximum_bytes=0))
    refuses("over_host_budget", lambda: HostRegistration([sources[0]], maximum_bytes=1), MemoryError)
    pinned = sources[0].pin_memory()
    refuses("foreign_pinned_not_owned", lambda: HostRegistration([pinned]))
    checks["foreign_pinned_still_owned_by_torch"] = pinned.is_pinned()
    source = sources[0].clone()
    bank = HostRegistration([source])
    source.add_(1)
    refuses("mutation_detected", lambda: bank.validate(source))
    refuses("mutation_closure_still_unregisters", bank.close)
    checks["mutation_failure_drained_and_unpinned"] = bank.closed and not source.is_pinned()
    bank.close()
    # A real registration followed by a controlled second registration failure must be drained.
    first, second = sources[0].clone(), sources[1].clone()
    cudart = torch.cuda.cudart()
    original_register = cudart.cudaHostRegister
    calls = []

    def fail_second(pointer, size, flags):
        calls.append(pointer)
        return original_register(pointer, size, flags) if len(calls) == 1 else 2

    cudart.cudaHostRegister = fail_second
    try:
        refuses("partial_registration_failure", lambda: HostRegistration([first, second]), RuntimeError)
    finally:
        cudart.cudaHostRegister = original_register
    checks["partial_real_first_registration_unregistered"] = len(calls) == 2 and not first.is_pinned() and not second.is_pinned()
    result = {"status": "generic_passed" if all(checks.values()) else "generic_failed",
              "checks": checks, "counted": False, "seconds": time.perf_counter()-started,
              "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("registered_camera_copy.py"))]}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    if not all(checks.values()):
        raise AssertionError("A registered image transfer ownership or exactness guard failed")


if __name__ == "__main__":
    main()

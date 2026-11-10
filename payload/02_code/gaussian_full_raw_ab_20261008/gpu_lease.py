"""Serialize heavy GPU stages through one Windows-owned byte lock and durable owner."""

from contextlib import contextmanager
import json
import math
import msvcrt
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity
from atomic_json_windows import atomic_json, read_json

SHARED = ROOT / "04_outputs/gaussian_full_raw_ab_20261008/gpu_coordination"
NATIVE_COLMAP = ROOT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"


def reboot_native_owner_evidence(owner):
    """Prove a native-only abandoned owner ended before this Windows boot."""
    if os.name != 'nt' or owner.get('status') != 'acquired':
        return None
    required = ('pid', 'process_birth', 'receipt', 'label', 'source', 'started_unix', 'acquired_unix')
    if any(key not in owner for key in required) or not isinstance(owner['source'], dict):
        return None
    source = Path(owner['source'].get('path', ''))
    if not source.is_file() or not source.samefile(Path(__file__)):
        return None
    boot = psutil.boot_time()
    born = owner.get('process_birth')
    if not isinstance(born, (int, float)) or not math.isfinite(born) or not 0 < born < boot:
        return None
    if any(not isinstance(owner[key], (int, float)) or not math.isfinite(owner[key])
            or not 0 < owner[key] < boot for key in ('started_unix', 'acquired_unix')):
        return None
    lease_path = Path(owner['receipt']).resolve(strict=True)
    lease = read_json(lease_path)
    for key in ('status', 'pid', 'process_birth', 'receipt', 'label', 'source', 'acquired_unix'):
        if lease.get(key) != owner.get(key):
            raise ValueError('The reboot owner and original GPU receipt disagree')
    commands = []
    identities = {(int(owner['pid']), float(born))}
    if lease.get('command') and lease.get('child_pid') is not None and lease.get('child_birth') is not None:
        commands = [lease['command']]
        identities.add((int(lease['child_pid']), float(lease['child_birth'])))
        identities.update((int(row['pid']), float(row['creation_time']))
            for row in lease.get('owned_process_identities', []))
        evidence_path = lease_path
    else:
        # Legacy SfM writes native commands and tracked children to its stage journal.
        evidence_path = lease_path.parent / 'STATE.json'
        if not evidence_path.is_file():
            return None
        journal = read_json(evidence_path)
        stages = [row for row in journal.get('stages', []) if row.get('name') == owner['label']]
        if journal.get('controller_pid') != owner['pid'] or len(stages) != 1:
            return None
        stage = stages[0]
        if stage.get('status') != 'running' or not stage.get('owned_births'):
            return None
        commands = stage.get('commands', [])
        identities.update((int(row['pid']), float(row['birth'])) for row in stage['owned_births'])
    if not commands or len(identities) < 2:
        return None
    for command in commands:
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            return None
        executable = Path(command[0])
        if not executable.is_file() or not executable.samefile(NATIVE_COLMAP):
            return None
    for pid, birth in identities:
        if pid <= 0 or not math.isfinite(birth) or not 0 < birth < boot:
            return None
        try:
            if psutil.Process(pid).create_time() == birth:
                return None
        except psutil.NoSuchProcess:
            pass
    return {'status': 'previous_windows_boot_native_GPU_owner_identity_verified_ended',
        'current_windows_boot_unix': boot, 'original_lease_receipt': str(lease_path),
        'native_command_evidence': str(evidence_path), 'native_commands': commands,
        'original_owner': owner, 'recorded_process_identities': [
            {'pid': pid, 'creation_time': birth} for pid, birth in sorted(identities)],
        'all_recorded_identities_before_current_boot': True,
        'all_recorded_identities_ended': True, 'native_COLMAP_only': True,
        'WSL_process_termination_inferred': False, 'original_process_exit_codes_known': False,
        'original_failed_or_interrupted_attempt_became_pass': False,
        'runtime_language_model_used': False, 'manual_intervention_used': False}


def recover_reboot_native_owner(owner_path, current_receipt):
    """Recover only while the caller holds the exclusive project GPU byte lock."""
    owner = read_json(owner_path)
    evidence = reboot_native_owner_evidence(owner)
    if evidence is None:
        return None
    destination = SHARED / ('HOST_REBOOT_NATIVE_OWNER_RECOVERY_' + str(time.time_ns()) + '.json')
    evidence.update(status='offline_previous_boot_native_GPU_owner_reclaimed',
        new_lease_receipt=str(Path(current_receipt).resolve()), exclusive_byte_lock_acquired=True,
        original_owner_and_lease_retained_in_evidence=True, original_lease_receipt_modified=False,
        recovered_unix=time.time())
    atomic_json(destination, evidence)
    atomic_json(owner_path, dict(owner, status='released', offline_reboot_recovery_receipt=str(destination),
        previous_boot_owner_verified_ended=True, released_unix=time.time()))
    return destination


def interrupted_native_owner_evidence(owner):
    """Require an ended native command and its actually failed owned supervisor."""
    if os.name != 'nt' or owner.get('status') != 'acquired':
        return None
    required = ('pid', 'process_birth', 'receipt', 'label', 'source', 'started_unix', 'acquired_unix')
    if any(key not in owner for key in required) or not isinstance(owner['source'], dict):
        return None
    source = Path(owner['source'].get('path', ''))
    if not source.is_file() or not source.samefile(Path(__file__)):
        return None
    boot, now = psutil.boot_time(), time.time()
    if any(not isinstance(owner[key], (int, float)) or not math.isfinite(owner[key])
            or not boot <= owner[key] <= now for key in ('process_birth', 'started_unix', 'acquired_unix')):
        return None
    lease_path = Path(owner['receipt']).resolve(strict=True)
    if not lease_path.name.endswith('_GPU_LEASE.json'):
        return None
    lease = read_json(lease_path)
    for key in ('status', 'pid', 'process_birth', 'receipt', 'label', 'source', 'acquired_unix'):
        if lease.get(key) != owner.get(key):
            raise ValueError('The interrupted owner and original GPU receipt disagree')
    if (lease.get('command_drain_verified') is not False or not lease.get('command')
            or lease.get('child_pid') is None or lease.get('child_birth') is None):
        return None
    command = lease['command']
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        return None
    executable = Path(command[0])
    if not executable.is_file() or not executable.samefile(NATIVE_COLMAP):
        return None
    lifecycle_path = lease_path.with_name(lease_path.name.removesuffix('_GPU_LEASE.json')
        + '_COMMAND_LIFECYCLE.json')
    if not lifecycle_path.is_file():
        return None
    lifecycle = read_json(lifecycle_path)
    if (lifecycle.get('status') != 'running' or lifecycle.get('command') != command
            or lifecycle.get('child_pid') != lease['child_pid'] or lifecycle.get('owned_tree_drained') is not False):
        return None
    pair = {(int(owner['pid']), float(owner['process_birth'])),
            (int(lease['child_pid']), float(lease['child_birth']))}
    native = {(int(row['pid']), float(row['creation_time']))
        for row in lifecycle.get('owned_process_identities', [])}
    if (int(lease['child_pid']), float(lease['child_birth'])) not in native:
        return None
    expected_guard = ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005/run_bounded_task.py'
    failed = []
    for parent in list(lease_path.parents)[:5]:
        candidates = set(parent.glob('*_process.json'))
        if (parent / 'PROCESS.json').is_file():
            candidates.add(parent / 'PROCESS.json')
        for path in sorted(candidates):
            value = read_json(path)
            guard_source = Path(value.get('source', {}).get('path', ''))
            identities = {(int(row['pid']), float(row['creation_time']))
                for row in value.get('owned_process_identities', [])}
            if (value.get('status') == 'failed_controller' and value.get('returncode') not in (None, 0)
                    and value.get('error') and guard_source.is_file()
                    and guard_source.samefile(expected_guard) and value['source'] == data_identity(expected_guard)
                    and pair <= identities):
                failed.append((path, value, identities))
    if not failed:
        return None
    guard_path, guard, identities = failed[0]
    identities = identities | pair | native
    for pid, birth in identities:
        if pid <= 0 or not math.isfinite(birth) or not boot <= birth <= now:
            return None
        try:
            if psutil.Process(pid).create_time() == birth:
                return None
        except psutil.NoSuchProcess:
            pass
    return {'status': 'same_boot_interrupted_native_GPU_owner_and_failed_supervisor_verified_ended',
        'current_windows_boot_unix': boot, 'original_lease_receipt': str(lease_path),
        'original_owner': owner, 'original_lease': lease,
        'native_command_evidence': str(lifecycle_path), 'original_native_lifecycle': lifecycle,
        'failed_guard_receipt': str(guard_path), 'original_failed_guard': guard,
        'native_commands': [command], 'recorded_process_identities': [
            {'pid': pid, 'creation_time': birth} for pid, birth in sorted(identities)],
        'all_recorded_identities_ended': True, 'all_recorded_identities_in_current_boot': True,
        'failed_supervisor_owned_native_parent_and_child': True, 'native_COLMAP_only': True,
        'WSL_process_termination_inferred': False, 'original_native_command_exit_code_known': False,
        'original_failed_or_interrupted_attempt_became_pass': False,
        'runtime_language_model_used': False, 'manual_intervention_used': False}


def recover_interrupted_native_owner(owner_path, current_receipt):
    """Preserve the interrupted lease while holding the exclusive project byte lock."""
    evidence = interrupted_native_owner_evidence(read_json(owner_path))
    if evidence is None:
        return None
    destination = SHARED / ('INTERRUPTED_NATIVE_OWNER_RECOVERY_' + str(time.time_ns()) + '.json')
    evidence.update(status='offline_interrupted_native_GPU_owner_reclaimed',
        new_lease_receipt=str(Path(current_receipt).resolve()), exclusive_byte_lock_acquired=True,
        original_lease_receipt_modified=False, original_failed_guard_modified=False,
        recovered_unix=time.time())
    atomic_json(destination, evidence)
    atomic_json(owner_path, dict(evidence['original_owner'], status='released',
        offline_interruption_recovery_receipt=str(destination), released_unix=time.time()))
    return destination


@contextmanager
def gpu_lease(receipt, label, maximum_wait_seconds=2400):
    receipt = Path(receipt)
    if receipt.exists() or maximum_wait_seconds <= 0:
        raise ValueError("A GPU stage lease requires a unique receipt and positive wait bound")
    SHARED.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    handle = (SHARED / "GPU.lock").open("a+b", buffering=0)
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"\0")
    state = {"status": "waiting", "label": label, "pid": os.getpid(),
        "process_birth": psutil.Process().create_time(), "source": data_identity(Path(__file__)),
        "receipt": str(receipt), "lock_scope": "Windows parents of native GPU commands and WSL GPU commands",
        "maximum_wait_seconds": maximum_wait_seconds, "started_unix": time.time()}
    atomic_json(receipt, state)
    started, acquired = time.monotonic(), False
    try:
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except OSError as error:
                if error.errno not in (13, 36) or time.monotonic()-started >= maximum_wait_seconds:
                    raise
                time.sleep(.25)
        owner = SHARED / "OWNER.json"
        if owner.exists() and read_json(owner)["status"] != "released":
            recovery = recover_reboot_native_owner(owner, receipt)
            if recovery is None:
                recovery = recover_interrupted_native_owner(owner, receipt)
                if recovery is None:
                    raise RuntimeError("A prior GPU owner did not drain; inspect its Windows and WSL children before reclaim")
                state['offline_interrupted_owner_recovery_receipt'] = str(recovery)
            else:
                state['offline_reboot_owner_recovery_receipt'] = str(recovery)
        state.update(status="acquired", wait_seconds=time.monotonic()-started, acquired_unix=time.time())
        atomic_json(receipt, state)
        atomic_json(owner, state)
        execution = time.monotonic()
        try:
            yield state
            state["child_result"] = "returned_without_exception"
        except BaseException as error:
            state["child_result"] = "exception_after_owned_command_drain"
            state["error"] = repr(error)
            raise
        finally:
            # Keep a failed drain quarantined even after the operating system unlocks the byte lock.
            status = "quarantined_owned_command_not_drained" if state.get("command_drain_verified") is False else "released"
            state.update(status=status, execution_seconds=time.monotonic()-execution, released_unix=time.time())
            atomic_json(receipt, state)
            atomic_json(owner, state)
    finally:
        if acquired:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def main():
    import argparse
    import subprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("An owned concrete GPU command is required")
    with gpu_lease(args.receipt, args.label) as state:
        child = subprocess.Popen(command)
        state.update(child_pid=child.pid, child_birth=psutil.Process(child.pid).create_time(), command=command)
        atomic_json(args.receipt, state)
        try:
            code = child.wait()
        except BaseException:
            child.wait()
            raise
        state["returncode"] = code
    raise SystemExit(code)


if __name__ == "__main__":
    main()

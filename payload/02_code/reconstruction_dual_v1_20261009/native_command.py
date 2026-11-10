"""Bound finite Windows commands while holding their GPU lease through cleanup."""

from contextlib import nullcontext
from pathlib import Path
import subprocess
import time
import traceback

import psutil

from common import atomic_json
from gpu_lease import gpu_lease
from process_ownership import capture_owned_tree


def live_owned(tracked):
    result = []
    for pid, born in tracked.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == born and process.is_running():
                result.append(process)
        except psutil.NoSuchProcess:
            pass
    return result


def drain_owned(tracked, child):
    """Use process birth identities so an unrelated reused PID cannot be killed."""
    owned = live_owned(tracked)
    for process in reversed(owned):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if child is not None:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
    psutil.wait_procs(owned, timeout=10)
    if live_owned(tracked):
        raise RuntimeError('The finite native command did not drain its owned process tree')


def run_native(command, output, label, maximum_seconds, gpu=False, *,
        lease_filename='GPU_LEASE.json', lease_label='supported_camera_recovery'):
    """Do not use this Windows process guard to infer that a WSL job has stopped."""
    command = list(map(str, command))
    if not command or maximum_seconds <= 0 or Path(command[0]).name.lower() == 'wsl.exe':
        raise ValueError('A finite native command and a positive time budget are required')
    if Path(lease_filename).name != lease_filename or not lease_filename.endswith('.json'):
        raise ValueError('A native GPU lease must use a local JSON filename')
    output = Path(output)
    process_receipt = output / (label + '_COMMAND_LIFECYCLE.json')
    if process_receipt.exists():
        raise FileExistsError('A native command requires unique lifecycle evidence')
    lease_receipt = output / lease_filename
    context = gpu_lease(lease_receipt, lease_label, maximum_wait_seconds=60) if gpu else nullcontext(None)
    before = time.monotonic()
    tracked, child = {}, None
    rejected_ownership = {}
    lifecycle = {'status': 'starting', 'command': command, 'maximum_seconds': maximum_seconds,
        'natural_exit_wait_maximum_seconds': 1.0, 'owned_tree_drained': False}
    with (output / (label + '.log')).open('x', encoding='utf-8') as stream:
        with context as state:
            atomic_json(process_receipt, lifecycle)
            if state is not None:
                state['command_drain_verified'] = False
            try:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    born = psutil.Process(child.pid).create_time()
                    tracked[child.pid] = born
                    if state is not None:
                        state.update(child_pid=child.pid, child_birth=born, command=command)
                        atomic_json(lease_receipt, state)
                except psutil.NoSuchProcess:
                    if child.poll() is None:
                        raise
                lifecycle.update(status='running', child_pid=child.pid,
                    owned_process_identities=[{'pid': pid, 'creation_time': born} for pid, born in tracked.items()])
                atomic_json(process_receipt, lifecycle)
                execution = time.monotonic()
                while child.poll() is None:
                    try:
                        owned, rejected = capture_owned_tree(child.pid, born)
                        for row in rejected:
                            rejected_ownership[(row['pid'], row['creation_time'])] = row
                        for process in owned:
                            tracked[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic() - execution > maximum_seconds:
                        raise subprocess.TimeoutExpired(command, maximum_seconds)
                    time.sleep(.1)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
                owned = live_owned(tracked)
                if owned:
                    # Windows launchers can return while their interpreter finishes kernel exit.
                    lifecycle.update(status='native_parent_exited_waiting_owned_exit',
                        owned_at_parent_exit=[{'pid': process.pid, 'creation_time': tracked[process.pid]}
                            for process in owned])
                    atomic_json(process_receipt, lifecycle)
                    remaining = max(0.0, maximum_seconds - (time.monotonic() - execution))
                    waiting = time.monotonic()
                    psutil.wait_procs(owned, timeout=min(1.0, remaining))
                    lifecycle['natural_exit_wait_seconds'] = time.monotonic() - waiting
                if live_owned(tracked):
                    raise RuntimeError('A finite native command left a live descendant after exiting')
                lifecycle.update(status='completed_native_command_owned_tree_drained', owned_tree_drained=True)
                if state is not None:
                    state.update(command_drain_verified=True, returncode=child.returncode)
            except BaseException:
                lifecycle.update(status='failed_native_command', traceback=traceback.format_exc())
                drain_owned(tracked, child)
                lifecycle['owned_tree_drained'] = True
                if state is not None:
                    state['command_drain_verified'] = True
                raise
            finally:
                lifecycle.update(elapsed_seconds=time.monotonic() - before,
                    ancestry_ownership_rejections=list(rejected_ownership.values()),
                    returncode=child.returncode if child is not None else None,
                    owned_process_identities=[{'pid': pid, 'creation_time': born} for pid, born in tracked.items()])
                atomic_json(process_receipt, lifecycle)
                if state is not None:
                    state['owned_process_identities'] = [
                        {'pid': pid, 'creation_time': born} for pid, born in tracked.items()]
    return time.monotonic() - before

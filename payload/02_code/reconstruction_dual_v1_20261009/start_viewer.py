"""Start or verify the owned local preview without opening an external window."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen

import psutil

from common import HERE, SFM_PYTHON, atomic_json, read_json
from process_ownership import capture_owned_tree


def stop_owned(root, record):
    try:
        parent = psutil.Process(record['pid'])
        if parent.create_time() != record['birth']:
            return
        expected_script, expected_root = str(HERE / 'serve_delivery.py'), str(root / 'viewer')
        command = parent.cmdline()
        if expected_script not in command or expected_root not in command:
            raise ValueError('The registered preview process ownership cannot be verified')
        owned, rejected = capture_owned_tree(record['pid'], record['birth'])
        children = [process for process in owned if process.pid != parent.pid]
        identities = []
        for process in [*children, parent]:
            try:
                command = process.cmdline()
                console_helper = (process.name().lower() == 'conhost.exe'
                    and Path(process.exe()).resolve() == (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve())
                if (expected_script not in command or expected_root not in command) and not console_helper:
                    raise ValueError('An unexpected descendant blocks preview shutdown')
                identities.append({'pid': process.pid, 'birth': process.create_time(),
                    'role': 'owned_Windows_console_helper' if console_helper else 'owned_preview_program'})
            except psutil.NoSuchProcess:
                pass
        atomic_json(root / ('PREVIEW_STOP_' + str(time.time_ns()) + '.json'),
            {'status': 'owned_preview_tree_shutdown_requested', 'identities': identities,
             'ancestry_ownership_rejections': rejected})
        for item in reversed(identities[:-1]):
            try:
                process = psutil.Process(item['pid'])
                if process.create_time() == item['birth']:
                    process.terminate()
                    process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
        if parent.is_running():
            parent.terminate()
            parent.wait(timeout=10)
    except psutil.NoSuchProcess:
        pass


def free_port():
    for port in range(8212, 8312):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                continue
        return port
    raise RuntimeError('No local preview port is available in the declared range')


def start(root, port=None):
    viewer = root / "viewer"
    identity = read_json(viewer / "DELIVERY_IDENTITY.json")
    receipt = root / "PREVIEW_SERVER.json"
    if receipt.is_file():
        record = read_json(receipt)
        try:
            process = psutil.Process(record["pid"])
            if process.create_time() == record["birth"]:
                with urlopen(record["url"] + "DELIVERY_IDENTITY.json", timeout=5) as response:
                    import json
                    if json.load(response) != identity:
                        raise ValueError("An owned preview returned another delivery")
                return record
        except psutil.NoSuchProcess:
            pass
    if port is None:
        port = free_port()
    command = [str(SFM_PYTHON), str(HERE / "serve_delivery.py"), "--root", str(viewer), "--port", str(port)]
    with (root / ("preview_server_" + str(time.time_ns()) + ".log")).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ, 'PYTHONPATH': str(HERE / 'offline_guard'), 'PIP_NO_INDEX': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
                'RECON_OFFLINE_DENIAL_LOG': str(root / 'PREVIEW_NETWORK_DENIALS.jsonl')})
    record = {"pid": process.pid, "birth": psutil.Process(process.pid).create_time(), "command": command,
              "url": "http://127.0.0.1:" + str(port) + "/", "run_root": str(root)}
    try:
        for _ in range(40):
            if process.poll() is not None:
                raise RuntimeError("The preview server exited before delivery verification")
            try:
                with urlopen(record["url"] + "DELIVERY_IDENTITY.json", timeout=1) as response:
                    import json
                    if json.load(response) != identity:
                        raise ValueError("The chosen preview port belongs to another delivery")
                owned, rejected = capture_owned_tree(record['pid'], record['birth'])
                record['owned_children'] = [{'pid': child.pid, 'birth': child.create_time()}
                    for child in owned if child.pid != record['pid']]
                record['ancestry_ownership_rejections'] = rejected
                atomic_json(receipt, record)
                return record
            except (ConnectionError, OSError):
                time.sleep(.25)
        raise RuntimeError("The owned preview did not become available")
    except BaseException:
        atomic_json(root / ('PREVIEW_START_FAILURE_' + str(time.time_ns()) + '.json'),
            {'status': 'failed_preview_start', 'record': record})
        stop_owned(root, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    print(start(args.run_root.resolve(), args.port)["url"])


if __name__ == "__main__":
    main()

"""Shared paths, durable records, and isolated delivery stage execution."""

import os
import hashlib
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
SFM_PYTHON = ROOT / "05_envs/sfm_py311_v1/Scripts/python.exe"
COLMAP = ROOT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
DISTRO = "E26-C2-LIV-20260907-134802"
WSL_PYTHON = "/opt/e26_thin_surface_v1/venv/bin/python"
sys.path.insert(0, str(OLD))
from atomic_json_windows import atomic_json, read_json


def importable_path(path):
    """Return a Windows extended path when imports may exceed MAX_PATH."""
    path = Path(path)
    value = str(path)
    if os.name == "nt" and path.is_absolute() and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def linux_path(path):
    value = Path(path).resolve().as_posix()
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("Expected an absolute Windows drive path")
    return "/mnt/" + value[0].lower() + value[2:]


def linux_command(script, *arguments):
    return ["wsl.exe", "-d", DISTRO, "--exec", "/usr/bin/env",
            "PYTHONPATH=" + linux_path(HERE / "offline_guard"), "PIP_NO_INDEX=1",
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "WANDB_MODE=disabled",
            "OMP_NUM_THREADS=8", "OPENBLAS_NUM_THREADS=8", WSL_PYTHON,
            linux_path(script), *map(str, arguments)]


def execute(root, label, command, budget, result, gpu=False):
    folder = Path(root) / "stages" / label
    folder.mkdir(parents=True, exist_ok=False)
    command = list(map(str, command))
    if "/usr/bin/env" in command and command[0] == "wsl.exe":
        command.insert(command.index(WSL_PYTHON), "RECON_OFFLINE_DENIAL_LOG=" + linux_path(folder / "NETWORK_DENIALS.jsonl"))
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    sources = []
    for source in sorted(HERE.rglob("*.py")):
        relative = source.relative_to(HERE)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources.append({"source": str(source), "snapshot": str(target),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    atomic_json(folder / "SOURCE_MANIFEST.json", {"sources": sources, "captured_before_stage": True})
    if gpu:
        command = [str(SFM_PYTHON), str(OLD / "gpu_lease.py"), "--receipt",
                   str(folder / "GPU_LEASE.json"), "--label", "dual_" + label,
                   "--", *map(str, command)]
    atomic_json(folder / "COMMAND.json", {"command": list(map(str, command)),
                "maximum_seconds": budget, "result": str(result)})
    guarded = [str(SFM_PYTHON), str(OLD / "run_owned_task.py"), "--receipt",
               str(folder / "PROCESS.json"), "--log", str(folder / "run.log"),
               "--maximum-seconds", str(budget), "--maximum-rss-gib", "32",
               "--", *map(str, command)]
    started = time.monotonic()
    completed = subprocess.run(guarded, env={**os.environ, "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE / "offline_guard"),
        "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled", "RECON_OFFLINE_DENIAL_LOG": str(folder / "NETWORK_DENIALS.jsonl")})
    measured = read_json(folder / "PROCESS.json")
    if completed.returncode or measured["status"] != "completed" or not Path(result).exists():
        raise RuntimeError("Delivery stage failed; retained evidence: " + str(folder))
    value = read_json(result)
    if value.get("status", "").startswith("failed"):
        raise RuntimeError("Stage result failed despite successful process: " + label)
    receipt = {"status": "completed", "label": label, "result": str(result),
               "controller_seconds": time.monotonic() - started, "process": measured}
    atomic_json(folder / "COMPLETED.json", receipt)
    return value

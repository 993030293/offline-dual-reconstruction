"""Drain only a private restore process identified by its saved birth and argv."""

import json
from pathlib import Path
import signal
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168")
    identity_file = output / "PROCESS_IDENTITY.json"
    if not identity_file.exists():
        raise RuntimeError("The Linux restore identity was not published; no unproven process is stopped")
    identity = json.loads(identity_file.read_text())
    if identity.get("scope") != "WSL":
        raise ValueError("The saved native process scope changed")

    def current():
        try:
            process = psutil.Process(identity["pid"])
            return process if process.is_running() and process.create_time() == identity["creation_time"] else None
        except psutil.NoSuchProcess:
            return None

    process = current()
    delivered = []
    if process is not None:
        command = process.cmdline()
        source = str(Path(__file__).with_name("profile_deferred_restore_equal_001.py"))
        if source not in command or "--output" not in command or Path(command[command.index("--output") + 1]).resolve() != output:
            raise ValueError("The exact owned restore argv no longer matches")
        for kind, duration in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
            process = current()
            if process is None:
                break
            process.send_signal(kind)
            delivered.append(int(kind))
            deadline = time.monotonic() + duration
            while current() is not None and time.monotonic() < deadline:
                time.sleep(.05)
    ended = current() is None
    print(json.dumps({"owned_identity": identity, "sent_signals": delivered, "saved_instance_ended": ended}), flush=True)
    if not ended:
        raise RuntimeError("The exact owned Linux restore did not drain")


if __name__ == "__main__":
    main()

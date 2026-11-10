"""Hold the campaign GPU lease for a bounded visible-browser audit."""

import argparse
from pathlib import Path
import time

from atomic_json_windows import atomic_json, read_json
from gpu_lease import gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=float, default=900)
    args = parser.parse_args()
    if not args.output.is_dir() or not 0 < args.maximum_seconds <= 900:
        raise ValueError("The browser audit needs a staged private directory and bounded execution")
    receipt = args.output / "GPU_LEASE.json"
    started = time.monotonic()
    with gpu_lease(receipt, "complete_current_model_browser_audit", maximum_wait_seconds=10):
        while not (args.output / "RESULT.json").exists():
            if time.monotonic() - started > args.maximum_seconds:
                raise TimeoutError("Unload the audit browser before any further GPU work; the visible audit did not return")
            time.sleep(.25)
        result = read_json(args.output / "RESULT.json")
        if result["status"] not in ("completed_outcomes_recorded", "failed_fixture"):
            raise ValueError("A partial browser audit cannot release its execution lease as complete")
        atomic_json(args.output / "LEASE_COMPLETION.json", {"status": "browser_finished_and_lease_released_on_return",
            "browser_status": result["status"], "browser_errors": result.get("errors", []),
            "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()

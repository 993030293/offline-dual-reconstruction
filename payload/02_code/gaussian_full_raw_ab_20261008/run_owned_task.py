"""Adapt only receipt publication while preserving the frozen process-tree guard."""

import importlib.util
from pathlib import Path
import sys
import time

from atomic_json_windows import atomic_json

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import data_identity


class OwnedReceiptPublisher:
    """Keep only proven heartbeat sharing conflicts outside algorithm failure."""

    def __init__(self, maximum_retry_seconds=2.0):
        self.running_published = False
        self.stats = {"successful_publications": 0, "sharing_retries": 0,
            "maximum_retry_seconds": maximum_retry_seconds,
            "adapter": data_identity(Path(__file__)),
            "publisher": data_identity(Path(__file__).with_name("atomic_json_windows.py")),
            "original_process_guard_unchanged": True, "deferred_heartbeat_publications": 0,
            "last_deferred_heartbeat_receipt": None, "terminal_publication_failure_is_fatal": True}

    def publish(self, path, state):
        path = Path(path)
        state["receipt_publication_before_current_write"] = dict(self.stats)
        started = time.monotonic()
        try:
            result = atomic_json(path, state,
                maximum_retry_seconds=self.stats["maximum_retry_seconds"])
        except OSError as error:
            evidence = getattr(error, "publication_sharing_evidence", {})
            eligible = (state.get("status") == "running" and self.running_published
                and bool(state.get("owned_process_identities"))
                and evidence.get("verified_delete_sharing_conflict") is True
                and Path(evidence["path"]).resolve() == path.resolve())
            if not eligible:
                raise
            record = path.with_name(path.name + ".HEARTBEAT_SHARING_" + str(time.time_ns()) + ".json")
            atomic_json(record, {"status": "offline_heartbeat_publication_deferred_final_receipt_required",
                "process_receipt": str(path.resolve()), "original_error": repr(error),
                "original_winerror": getattr(error, "winerror", None),
                "actual_sharing_evidence": evidence, "owned_process_identities": state["owned_process_identities"],
                "elapsed_seconds_before_publication": state["elapsed_seconds"],
                "publication_wait_seconds": time.monotonic() - started,
                "maximum_seconds_unchanged": state["maximum_seconds"],
                "maximum_RSS_bytes_unchanged": state["maximum_RSS_bytes"],
                "previous_complete_running_receipt_exists": True,
                "final_receipt_publication_still_required": True, "adapter": self.stats["adapter"],
                "publisher": self.stats["publisher"], "runtime_language_model_used": False,
                "manual_intervention_used": False, "created_unix": time.time()})
            self.stats["deferred_heartbeat_publications"] += 1
            self.stats["last_deferred_heartbeat_receipt"] = str(record.resolve())
            if state["elapsed_seconds"] + time.monotonic() - started > state["maximum_seconds"]:
                raise TimeoutError("The original task budget expired during heartbeat publication")
            return
        self.stats["successful_publications"] += 1
        self.stats["sharing_retries"] += result["retries"]
        if state.get("status") == "running":
            self.running_published = True


def main():
    spec = importlib.util.spec_from_file_location("frozen_raw_campaign_guard", V1 / "run_bounded_task.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.atomic_json = OwnedReceiptPublisher().publish
    guard.main()


if __name__ == "__main__":
    main()

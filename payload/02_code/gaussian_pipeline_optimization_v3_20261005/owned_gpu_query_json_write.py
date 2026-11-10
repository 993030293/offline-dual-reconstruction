"""Bound transient atomic replacement retries to one owned output tree."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
import runtime

ORIGINAL = runtime.atomic_json


class OwnedAtomicJSON:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001")
        self.trace = {"retry_deadline_seconds": .5, "setup_seconds": 0., "restore_seconds": 0.,
            "events": [], "binding_restored": False, "closed": False, "not_an_optimization": True}

    def __call__(self, path, payload):
        path = Path(path)
        owned = path.resolve().is_relative_to(self.root)
        if not owned:
            return ORIGINAL(path, payload)
        started, attempts, first = time.perf_counter(), 0, None
        try:
            while True:
                attempts += 1
                try:
                    return ORIGINAL(path, payload)
                except PermissionError as error:
                    if error.winerror != 5:
                        raise
                    if first is None:
                        first = {"type": type(error).__name__, "winerror": error.winerror, "message": str(error)}
                    remaining = .5-(time.perf_counter()-started)
                    if remaining <= 0:
                        raise
                    time.sleep(min(.01, remaining))
        finally:
            if first is not None:
                event = {"target": str(path.resolve()), "attempts": attempts, "first_error": first,
                    "complete_write_seconds": time.perf_counter()-started}
                self.trace["events"].append(event)
                folder = self.root/"write_recovery_events"
                folder.mkdir(exist_ok=True)
                with (folder/f"event_{len(self.trace['events']):06d}.json").open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, indent=2, allow_nan=False)+"\n")


def local_modules():
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source and Path(source).resolve().is_relative_to(ROOT/"02_code"):
            yield module


@contextmanager
def owned_atomic_json(root):
    started = time.perf_counter()
    operator = OwnedAtomicJSON(root)
    for module in local_modules():
        if vars(module).get("atomic_json") is ORIGINAL:
            module.atomic_json = operator
    operator.trace["setup_seconds"] = time.perf_counter()-started
    try:
        yield operator.trace
    finally:
        started = time.perf_counter()
        for module in local_modules():
            if vars(module).get("atomic_json") is operator:
                module.atomic_json = ORIGINAL
        operator.trace.update(binding_restored=runtime.atomic_json is ORIGINAL
            and not any(vars(m).get("atomic_json") is operator for m in local_modules()), closed=True,
            restore_seconds=time.perf_counter()-started)

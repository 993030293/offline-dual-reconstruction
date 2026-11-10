"""Compose the existing bounded identity owner with the exact R153 consumer."""

import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import threading
import time

import profile_fused_camera_fingerprint as parent

ROOT, CODE = parent.ROOT, parent.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
HELPER = CODE / "parallel_b_005/ordered_identity.py"


def transform_generated_scene(source):
    replacements = (
        ('sources = [data_identity(path) for path in source_paths]',
         'sources = _identity_phase(source_paths, data_identity, "sources_before")'),
        ('inputs = [data_identity(path) for path in sorted(dataset.rglob("*")) if path.is_file()]',
         'inputs = _identity_phase([path for path in sorted(dataset.rglob("*")) if path.is_file()], data_identity, "inputs_before")'),
        ('inputs == [data_identity(row["path"]) for row in inputs]',
         'inputs == _identity_phase([row["path"] for row in inputs], data_identity, "inputs_after")'),
        ('sources == [data_identity(path) for path in source_paths]',
         'sources == _identity_phase(source_paths, data_identity, "sources_after")'),
    )
    result = source
    for before, after in replacements:
        if result.count(before) != 1:
            raise ValueError("An original complete identity boundary changed")
        result = result.replace(before, after)
    return result


def main():
    index = sys.argv.index("--identity-enabled")
    enabled = sys.argv[index + 1]
    if enabled not in ("0", "1"):
        raise ValueError("The existing file identity route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    if sys.argv[sys.argv.index("--fused-enabled") + 1] != "1":
        raise ValueError("Both composed sides must use the same accepted image digest route")
    enabled = enabled == "1"
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists():
        raise FileExistsError("The composed complete Scene requires a fresh output")
    spec = importlib.util.spec_from_file_location("owned_Q68_identity", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    phases = []

    def identity_phase(paths, native, label):
        if not isinstance(paths, (list, tuple)):
            raise TypeError("The original complete ordered path list must remain owned")
        row = {"phase": label, "enabled": enabled, "paths": [str(Path(p)) for p in paths],
            "expected": len(paths), "native_module": native.__module__, "native_name": native.__name__}
        phases.append(row)
        started = time.perf_counter()
        trace = {}
        row["queue"] = trace
        try:
            values = helper.collect_identities(paths, native, trace) if enabled else [native(p) for p in paths]
            if len(values) != len(paths) or any(set(value) != {"path", "bytes", "sha256"}
                    or value["path"] != str(Path(path)) for path, value in zip(paths, values)):
                raise ValueError("The original complete identity schema or mapping changed")
            row["returned"] = len(values)
            row["status"] = "complete"
            return values
        except BaseException as error:
            row.update(status="failed", error=repr(error))
            raise
        finally:
            row["actual_phase_seconds_not_whole_acceptance"] = time.perf_counter() - started
            row["owned_identity_threads_closed"] = not any(t.name.startswith("b005-native-identity") for t in threading.enumerate())

    original = CODE / "profile_fused_camera_fingerprint.py"
    before = original.read_text(encoding="utf-8")
    after = before
    replacements = (
        ('output.relative_to(OUT.resolve())', 'output.relative_to(_combo_out.resolve())'),
        ('scope.replace("mi001", "scf001")', 'scope.replace("mi001", "scfq001")'),
        ('    sources = [Path(__file__), original,',
         '    after = _transform_generated_scene(after)\n    sources = [Path(__file__), original,'),
        ('    ids = [data_identity(path) for path in sources]',
         '    sources.extend(_combo_sources)\n    ids = [data_identity(path) for path in sources]'),
        ('    exec(compile(after, str(original), "exec"), namespace)',
         '    namespace["_identity_phase"] = _identity_phase\n    exec(compile(after, str(original), "exec"), namespace)'),
    )
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError("The accepted R153 wrapper source boundary changed")
        after = after.replace(old, new)
    archive = output.with_name(output.name + "_q68_sources")
    archive.mkdir(exist_ok=False)
    (archive / "EXECUTED_R153_PARENT.py").write_text(after, encoding="utf-8")
    (archive / "R153_PARENT.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="accepted_R153", tofile="owned_Q68_composition")), encoding="utf-8")
    shutil.copyfile(Path(__file__), archive / Path(__file__).name)
    shutil.copyfile(HELPER, archive / HELPER.name)
    namespace = {"__file__": str(original), "__name__": "accepted_R153_owned_Q68_composition",
        "_combo_out": OUT, "_transform_generated_scene": transform_generated_scene,
        "_combo_sources": [Path(__file__), HELPER, CODE / "q68_fused_fingerprint_node.py"],
        "_identity_phase": identity_phase}
    exec(compile(after, str(original), "exec"), namespace)
    try:
        namespace["main"]()
    except BaseException as error:
        if output.is_dir():
            parent.atomic_json(output / "FAILED_IDENTITY_TRACE.json", {"phases": phases, "error": repr(error), "new_optimization_count": 0})
        raise
    checks = {
        "all_original_four_identity_phases": [p["phase"] for p in phases] == ["sources_before", "inputs_before", "inputs_after", "sources_after"],
        "complete_original17_source_and1036_file_denominators": [p["returned"] for p in phases] == [17, 1036, 1036, 17],
        "original_native_identity_call_object": all(p["native_module"] == "runtime" and p["native_name"] == "data_identity" for p in phases),
        "all_before_after_original_path_orders_equal": phases[0]["paths"] == phases[3]["paths"] and phases[1]["paths"] == phases[2]["paths"],
        "all_identity_workers_drained_bounded4_queue8": all(p["owned_identity_threads_closed"] and (not enabled or
            p["queue"]["closed"] and p["queue"]["all_started_joined"] and p["queue"]["maximum_pending"] <= 8
            and p["queue"]["workers"] == 4 and p["queue"]["consumed"] == p["returned"] and not p["queue"]["worker_errors"]) for p in phases),
    }
    parent.atomic_json(output / "IDENTITY_TRACE.json", {"checks": checks, "phases": phases,
        "new_optimization_count": 0, "existing_Q68_R153_composition_only": True})
    if not all(checks.values()):
        raise AssertionError("The actual complete identity composition changed an original phase")


if __name__ == "__main__":
    main()

"""Complete actual failure, signal, and fresh recovery checks in new folders."""

from pathlib import Path
import shutil
import subprocess
import time

import psutil
import run_coordinate_campaign as common
from runtime import atomic_json, data_identity

OUT = common.OUT / "v0"


def run(label, mode, fault=None):
    leaf, node = OUT / label, OUT / (label + "_node")
    cmd = common.command(leaf, node, "profile_coordinate_failure.py" if fault else "profile_coordinate_scene.py", mode)
    if fault:
        cmd.extend(["--fault-kind", fault])
    started = time.perf_counter()
    with (OUT / (label + ".log")).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        code = process.wait(timeout=380)
    row = {"owner": owner, "actual_spawn_to_exit_seconds": time.perf_counter() - started,
           "returncode": code, "leaf": str(leaf), "node": str(node), "command": cmd}
    atomic_json(OUT / (label + "_EXIT.json"), row)
    if code:
        raise RuntimeError("A registered validation node failed: " + label)
    controller = common.read(node / "NODE.json")
    if (controller["status"] != "completed" or controller["live_Windows_after_wait"]
            or any(item["alive"] for item in controller["actual_Linux_after_wait"])
            or not controller["Linux_resource_receipts"]):
        raise AssertionError("A validation native owner did not close")
    print({"completed": label, "actual_seconds": row["actual_spawn_to_exit_seconds"]}, flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    old = common.read(common.OUT / "c0/RESULT.json")
    if old["status"] != "three_complete_coordinate_pairs_pending_validation0":
        raise ValueError("Complete frozen formal pairs must pass before more validation")
    sources = [Path(__file__), common.CODE / "profile_coordinate_failure.py",
               common.CODE / "native_coordinate_reuse.py", common.CODE / "run_coordinate_campaign.py"]
    identities = [data_identity(path) for path in sources]
    if old["sources"] != [data_identity(Path(row["path"])) for row in old["sources"]]:
        raise ValueError("The formal native sources changed")
    for pair in old["pairs"]:
        root = common.OUT / "c0" / ("t" + str(pair["index"]))
        common.compare(root / "incumbent", root / "candidate")
    archive = OUT / "source_snapshot"
    archive.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
    rows, checks, science = {}, {}, {}
    for fault in ("error", "sigint"):
        values = {}
        for mode in ("incumbent", "candidate"):
            label = fault + "_" + mode
            rows[label] = run(label, mode, fault)
            values[mode] = common.read(Path(rows[label]["leaf"]) / "FAULT_RESULT.json")
        science[fault] = values
        checks[fault + "_full_original_prefix_distance_input_error_and_drain_exact"] = (
            all(all(value["checks"].values()) for value in values.values())
            and values["incumbent"]["error"] == values["candidate"]["error"]
            and values["incumbent"]["evidence"] == values["candidate"]["evidence"]
            and values["incumbent"]["inputs"] == values["candidate"]["inputs"])
        atomic_json(OUT / "STATE.json", {"rows": rows, "checks": checks, "new_optimization_count": 0})
        if not checks[fault + "_full_original_prefix_distance_input_error_and_drain_exact"]:
            raise AssertionError("The complete native failure or signal contract changed")
    private = OUT / "private_recovery_copy"
    private.mkdir()
    removed = []
    for name in ("input.ply", "cameras.json"):
        source = Path(rows["error_candidate"]["leaf"]) / "scene" / name
        destination = private / name
        shutil.copyfile(source, destination)
        if destination.resolve().parent != private.resolve() or destination.is_symlink():
            raise ValueError("An owned derived recovery copy escaped its folder")
        identity = data_identity(destination)
        destination.unlink()
        removed.append({"private_copy": identity, "removed": not destination.exists(),
                        "failed_original_retained": source.is_file()})
    rows["recovered"] = run("recovered", "candidate")
    comparison = common.compare(common.OUT / "c0/t0/incumbent", Path(rows["recovered"]["leaf"]))
    checks.update(fresh_complete1016_recovery_all_original_contracts=all(comparison["checks"].values()),
                  only_two_owned_derived_copies_removed=len(removed) == 2
                    and all(row["removed"] and row["failed_original_retained"] for row in removed),
                  all_formal_and_validation_sources_unchanged=identities == [data_identity(path) for path in sources]
                    and old["sources"] == [data_identity(Path(row["path"])) for row in old["sources"]],
                  family_disk_resource_bounds=sum(path.stat().st_size for path in common.OUT.rglob("*") if path.is_file()) <= 2 * 1024**3
                    and shutil.disk_usage(OUT).free >= 20 * 1024**3)
    result = {"status": "complete_validation_passed_pending_separate_review0" if all(checks.values()) else "failed",
              "rows": rows, "checks": checks, "sources": identities, "removed": removed,
              "science": science, "comparison": comparison, "new_optimization_count": 0}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(OUT / "STATE.json", result)
    if not all(checks.values()):
        raise AssertionError("The complete coordinate validation failed")


if __name__ == "__main__":
    main()

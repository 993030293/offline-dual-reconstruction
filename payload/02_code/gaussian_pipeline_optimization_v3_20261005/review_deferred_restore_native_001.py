"""Verify current native lineage and closure after all restore diagnostics end."""

import hashlib
import json
from pathlib import Path
import sys

import psutil


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def local(path):
    path = path.replace("\\", "/")
    if len(path) > 2 and path[1:3] == ":/":
        return Path("/mnt/" + path[0].lower() + "/" + path[3:])
    return Path(path)


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(BASE)
    output.mkdir(exist_ok=False)
    identities, saved, copies = {}, {}, []
    contracts = list((BASE.parent).rglob("CONTRACT.json"))
    for path in contracts:
        body = read(path)
        for row in body.get("sources", []) + body.get("inputs", []):
            key = (row["path"], row["sha256"], row["bytes"])
            identities[key] = row
        archive = path.parent / "source_copies"
        if archive.exists():
            for index, row in enumerate(body["sources"]):
                copies.append((archive / (f"{index:02d}_" + local(row["path"]).name), row))
    for folder in ("generic0", "generic2"):
        for row in read(BASE / folder / "RESULT.json")["sources_inputs"]:
            identities[(row["path"], row["sha256"], row["bytes"])] = row
    checked = []
    cache = {}
    for row in identities.values():
        path = local(row["path"])
        if path not in cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024**2), b""):
                    digest.update(block)
            cache[path] = (path.stat().st_size, digest.hexdigest())
        size, digest = cache[path]
        checked.append({"path": row["path"], "exact": size == row["bytes"] and digest == row["sha256"]})
    copy_checks = [path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        for path, row in copies]
    for scope in (BASE.parent, ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup013"):
        for name in ("PROCESS_IDENTITY.json", "WSL_RESOURCE.json", "CPU_BOOTSTRAP_RESOURCE.json"):
            for path in scope.rglob(name):
                row = read(path)
                if row.get("scope") == "WSL" and "pid" in row and "creation_time" in row:
                    saved[(row["pid"], row["creation_time"])] = row
    ended = []
    for (pid, born), row in saved.items():
        try:
            process = psutil.Process(pid)
            closed = process.create_time() != born or not process.is_running()
        except psutil.NoSuchProcess:
            closed = True
        ended.append({"pid": pid, "creation_time": born, "saved_instance_ended": closed})
    resource = [read(path) for path in BASE.parent.rglob("WSL_RESOURCE.json")]
    resource += [read(path) for path in BASE.parent.rglob("CPU_BOOTSTRAP_RESOURCE.json")]
    checks = {"all_current_native_source_and_data_identities_exact": bool(checked) and all(row["exact"] for row in checked),
        "all_complete_native_source_copies_exact": bool(copy_checks) and all(copy_checks),
        "all_saved_native_birth_instances_ended": bool(ended) and all(row["saved_instance_ended"] for row in ended),
        "all_native_RSS_and_time_limits_kept": all(row["peak_tree_RSS_bytes"] <= row["maximum_RSS_bytes"]
            and row["elapsed_seconds"] <= row["maximum_seconds"] + .2 for row in resource)}
    result = {"checks": checks, "unique_expected_source_and_input_rows": len(checked),
        "unique_current_paths": len(cache), "complete_source_copies": len(copies), "native_saved_instances": ended,
        "resource_records": len(resource), "writer_identity": {"scope": "WSL", "pid": psutil.Process().pid,
            "creation_time": psutil.Process().create_time()}, "new_optimization_count": 0,
        "hashes_are_code_data_lineage_not_document_acceptance": True}
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "identities": len(checked), "copies": len(copies), "native_births": len(ended)}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The native source, input or lifecycle review failed")


if __name__ == "__main__":
    main()

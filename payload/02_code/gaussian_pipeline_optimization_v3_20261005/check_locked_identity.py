"""Test immutable Windows hashing on owned files, aliases and SQLite readers."""

import argparse
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from locked_input_identity import LockedInputs


def rejected(function):
    try:
        value = function()
        if hasattr(value, "close"):
            value.close()
    except OSError:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    owned = args.output/"owned"
    owned.mkdir()
    path = owned/"input.bin"
    path.write_bytes(bytes(range(256))*1234)
    alias = owned/"alias.bin"
    alias.hardlink_to(path)
    db = owned/"input.db"
    connection = sqlite3.connect(db)
    try:
        connection.execute("CREATE TABLE payload(value BLOB)")
        connection.execute("INSERT INTO payload VALUES(?)", (b"immutable native rows",))
        connection.commit()
    finally:
        connection.close()
    snapshot = LockedInputs([owned], data_identity)
    try:
        expected = data_identity(path)
        first, second = snapshot.identity(path), snapshot.identity(path)
        snapshot.identity(db)
        connection = sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT value FROM payload").fetchone()
        finally:
            connection.close()
        checks = {"complete_native_hash_and_original_path_exact": first == second == expected,
            "write_rejected": rejected(lambda: path.open("r+b")),
            "truncation_rejected": rejected(lambda: path.open("wb")),
            "delete_rejected": rejected(path.unlink), "rename_rejected": rejected(lambda: path.rename(owned/"replacement.bin")),
            "hardlink_alias_write_rejected": rejected(lambda: alias.open("r+b")),
            "readonly_SQLite_still_works": row == (b"immutable native rows",)}
        unix_path = "/mnt/c/"+path.resolve().as_posix()[3:]
        script = "import sys\ntry:\n f=open(sys.argv[1],'r+b');f.write(b'owned WSL write probe');f.close()\nexcept OSError:\n sys.exit(7)\n"
        probe = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
            "/opt/e26_thin_surface_v1/venv/bin/python", "-c", script, unix_path], capture_output=True, text=True, timeout=20)
        checks["owned_WSL_DrvFS_write_also_rejected"] = probe.returncode == 7 and data_identity(path) == expected
        snapshot.verify()
        receipt = snapshot.receipt()
    finally:
        snapshot.close()
    with path.open("r+b") as stream:
        stream.write(b"after explicit release")
    checks["owned_file_writable_after_close_and_digest_changed"] = data_identity(path) != expected
    held = path.open("r+b")
    try:
        blocked = LockedInputs([owned], data_identity)
        try:
            checks["preexisting_writer_rejects_snapshot_acquisition"] = rejected(lambda: blocked.identity(path))
        finally:
            blocked.close()
    finally:
        held.close()
    result = {"status": "owned_Windows_snapshot_guards_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "snapshot": receipt, "original_inputs_touched": False, "counted": False,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("locked_input_identity.py"))]}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks")})
    if not all(checks.values()):
        raise ValueError("A byte immutability, full hash or ownership guard failed")


if __name__ == "__main__":
    main()

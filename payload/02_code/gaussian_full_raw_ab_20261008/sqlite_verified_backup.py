"""Preserve a closed SQLite source bundle and verify every backed-up native value."""

from pathlib import Path
import shutil
import sqlite3


def complete_bytes_equal(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while block := first.read(4 * 1024**2):
            if block != second.read(len(block)):
                return False
        return not second.read(1)


def verified_private_backup(source, target, scratch, identity):
    source, target, scratch = map(Path, (source, target, scratch))
    if target.exists() or not source.is_file() or source.is_symlink():
        raise ValueError("Verified database restoration requires an ordinary source and unique target")
    scratch.mkdir(parents=True, exist_ok=False)
    bundle = [path for path in (source, source.with_name(source.name + "-wal"),
                               source.with_name(source.name + "-shm")) if path.exists()]
    signatures = [identity(path) for path in bundle]
    for path in bundle:
        shutil.copyfile(path, scratch / path.name)
        if not complete_bytes_equal(path, scratch / path.name):
            raise AssertionError("A complete SQLite source bundle member changed while copying")
    private_source = scratch / source.name
    tables = []
    with sqlite3.connect(private_source.as_uri() + "?mode=ro", uri=True) as original:
        if original.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The complete private SQLite bundle failed integrity checking")
        with sqlite3.connect(target) as restored:
            original.backup(restored)
            restored.execute("PRAGMA journal_mode=DELETE")
            if restored.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("The full backed-up database failed integrity checking")
            schema_query = "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"
            schema = original.execute(schema_query).fetchall()
            if schema != restored.execute(schema_query).fetchall():
                raise AssertionError("The complete native table schemas changed during backup")
            for name, _ in schema:
                escaped = '"' + name.replace('"', '""') + '"'
                columns = original.execute("PRAGMA table_info(" + escaped + ")").fetchall()
                if columns != restored.execute("PRAGMA table_info(" + escaped + ")").fetchall():
                    raise AssertionError("The complete column definitions changed")
                keys = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
                order = ",".join('"' + key.replace('"', '""') + '"' for key in keys) if keys else "rowid"
                query = "SELECT * FROM " + escaped + " ORDER BY " + order
                first, second = original.execute(query), restored.execute(query)
                count = 0
                while True:
                    a, b = first.fetchmany(32), second.fetchmany(32)
                    if a != b:
                        raise AssertionError("A full native SQLite row or blob differs: " + name)
                    count += len(a)
                    if not a:
                        break
                tables.append({"table": name, "rows": count, "all_values_and_blobs_literal_equal": True})
    if signatures != [identity(path) for path in bundle]:
        raise ValueError("An archived original SQLite bundle changed during restoration")
    return {"route": "SQLite_online_backup_from_complete_private_bundle", "source_members": signatures,
            "tables": tables, "source_bundle_unchanged": True, "all_schema_rows_and_blobs_exact": True,
            "original_WAL_bytes": source.with_name(source.name + "-wal").stat().st_size
                if source.with_name(source.name + "-wal").exists() else 0}

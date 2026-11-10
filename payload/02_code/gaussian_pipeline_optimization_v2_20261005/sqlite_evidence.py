"""Compare closed SQLite evidence while checking its explicit header counters."""

from pathlib import Path
import hashlib
import sqlite3


def compare_closed_databases(reference, candidate, mapper_calls):
    """Allow only synchronized counter advances; all payload bytes stay exact."""
    reference, candidate = Path(reference), Path(candidate)
    for path in (reference, candidate):
        wal = Path(str(path)+"-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Database evidence still has an uncheckpointed WAL")
    if reference.stat().st_size != candidate.stat().st_size or mapper_calls < 0:
        raise ValueError("Database size or mapper-call bound differs")
    count = 0
    with reference.open("rb") as first, candidate.open("rb") as second:
        a, b = bytearray(first.read(100)), bytearray(second.read(100))
        if a[:16] != b"SQLite format 3\0" or b[:16] != a[:16]:
            raise ValueError("Invalid SQLite header")
        before = int.from_bytes(a[24:28], "big")
        after = int.from_bytes(b[24:28], "big")
        if a[24:28] != a[92:96] or b[24:28] != b[92:96]:
            raise ValueError("SQLite validity and transaction counters disagree")
        delta = (after-before) % 2**32
        if not 0 <= delta <= mapper_calls:
            raise ValueError("Unexpected number of database transaction advances")
        for start in (24, 92):
            a[start:start+4] = b"\0"*4
            b[start:start+4] = b"\0"*4
        if a != b:
            raise ValueError("Database header changed beyond transaction metadata")
        count = len(a)
        while True:
            a, b = first.read(4*1024**2), second.read(4*1024**2)
            if a != b:
                raise ValueError("Database evidence payload changed")
            if not a:
                break
            count += len(a)
    return {"payload_and_other_header_bytes_exact": True, "bytes_compared": count,
            "counter_before": before, "counter_after": after, "mapper_calls_bound": mapper_calls,
            "counter_increment": delta, "ignored_payload_bytes": 0}


def compare_database_content(reference, candidate):
    """Use SQLite read transactions to include committed WAL evidence exactly."""
    paths = [Path(reference).resolve(), Path(candidate).resolve()]
    connections = [sqlite3.connect("file:"+path.as_posix()+"?mode=ro", uri=True) for path in paths]
    try:
        for connection in connections:
            connection.execute("BEGIN")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite evidence integrity check failed")
        schemas = [connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
                   for connection in connections]
        if schemas[0] != schemas[1]:
            raise ValueError("SQLite evidence schemas differ")
        for pragma in ("user_version", "application_id", "encoding", "page_size"):
            if connections[0].execute("PRAGMA "+pragma).fetchall() != connections[1].execute("PRAGMA "+pragma).fetchall():
                raise ValueError("SQLite evidence format contract differs")
        tables = []
        for kind, name, _, _ in schemas[0]:
            if kind != "table":
                continue
            quoted = '"'+name.replace('"', '""')+'"'
            fields = connections[0].execute("PRAGMA table_info("+quoted+")").fetchall()
            primary = sorted((row[5], row[1]) for row in fields if row[5])
            order_names = [name for _, name in primary] if primary else [row[1] for row in fields]
            ordering = ",".join('"'+column.replace('"', '""')+'"' for column in order_names)
            query = "SELECT * FROM "+quoted+" ORDER BY "+ordering
            cursors = [connection.execute(query) for connection in connections]
            rows, blob_bytes = 0, 0
            digest = hashlib.sha256()
            while True:
                a, b = cursors[0].fetchone(), cursors[1].fetchone()
                if a != b:
                    raise ValueError("SQLite evidence row differs in "+name+" at ordinal "+str(rows))
                if a is None:
                    break
                for value in a:
                    payload = value if isinstance(value, bytes) else repr(value).encode("utf-8")
                    tag = type(value).__name__.encode("ascii")
                    digest.update(len(tag).to_bytes(4,"big")+tag+len(payload).to_bytes(8,"big")+payload)
                    blob_bytes += len(payload) if isinstance(value, bytes) else 0
                rows += 1
            tables.append({"table": name, "rows": rows, "blob_bytes_compared": blob_bytes, "content_sha256": digest.hexdigest()})
        return {"all_schema_and_rows_and_blobs_exact": True, "committed_WAL_included": True, "tables": tables,
                "content_bytes_ignored": 0, "physical_transaction_counters_are_not_feature_evidence": True}
    finally:
        for connection in connections:
            connection.rollback()
            connection.close()

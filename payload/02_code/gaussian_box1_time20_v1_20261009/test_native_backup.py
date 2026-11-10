"""Verify WAL-aware preservation and rejection of changed native database values."""

from contextlib import closing
import gc
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from audit_incremental_matching import complete_native_tables_equal, data_identity, verified_private_backup


class NativeBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temporary.name)

    def populate(self, path, value=b"complete-native-blob"):
        with closing(sqlite3.connect(path)) as connection:
            with connection:
                connection.execute("CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, data BLOB)")
                connection.execute("INSERT INTO descriptors VALUES(?,?)", (1, value))

    def test_complete_committed_WAL_values_preserved(self):
        source, target = self.root / "source.db", self.root / "verified.db"
        with closing(sqlite3.connect(source)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, data BLOB)")
            writer.execute("INSERT INTO descriptors VALUES(?,?)", (1, b"committed-in-WAL"))
            writer.commit()
            self.assertGreater(source.with_name(source.name+"-wal").stat().st_size, 0)
            incomplete = self.root / "main-file-only.db"
            shutil.copyfile(source, incomplete)
            with closing(sqlite3.connect(incomplete.as_uri()+"?immutable=1", uri=True)) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM sqlite_master WHERE name='descriptors'").fetchone()[0], 0)
            result = verified_private_backup(source, target, self.root / "complete_bundle", data_identity)
            self.assertTrue(result["source_bundle_unchanged"])
            self.assertTrue(result["all_schema_rows_and_blobs_exact"])
            with closing(sqlite3.connect(target)) as connection:
                self.assertEqual(connection.execute("SELECT * FROM descriptors").fetchall(), [(1, b"committed-in-WAL")])

    def test_changed_complete_blob_rejected(self):
        first, second = self.root / "first.db", self.root / "second.db"
        self.populate(first); self.populate(second, b"changed-native-blob")
        with self.assertRaisesRegex(AssertionError, "native row or blob"):
            complete_native_tables_equal(first, second)

    def test_equal_complete_tables_accepted(self):
        first, second = self.root / "first.db", self.root / "second.db"
        self.populate(first); self.populate(second)
        self.assertEqual(complete_native_tables_equal(first, second), [
            {"table": "descriptors", "complete_rows": 1, "all_values_and_blobs_equal": True}])


if __name__ == "__main__":
    unittest.main()

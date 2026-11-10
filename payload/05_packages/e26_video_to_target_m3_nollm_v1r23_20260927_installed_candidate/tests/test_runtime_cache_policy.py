"""Check installed runtime caches without weakening source package hygiene."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "verify_package.py"
SPEC = importlib.util.spec_from_file_location("verify_package", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeCachePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "code" / "__pycache__"
        self.cache.mkdir(parents=True)
        (self.cache / "example.cpython-311.pyc").write_bytes(b"cache")

    def install_receipt(self, status: str) -> None:
        (self.root / "INSTALL_RECEIPT.json").write_text(
            json.dumps({"schema_version": "e26_install_receipt_v1", "status": status}),
            encoding="utf-8",
        )

    def test_source_package_with_cache_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "compiled Python cache"):
            MODULE.validate_runtime_caches(self.root)

    def test_verified_install_can_be_checked_after_runtime_imports(self) -> None:
        self.install_receipt("installed_and_runtime_verified")
        self.assertEqual(MODULE.validate_runtime_caches(self.root), 1)

    def test_non_bytecode_cache_content_is_rejected(self) -> None:
        self.install_receipt("installed_and_runtime_verified")
        (self.cache / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "compiled Python cache"):
            MODULE.validate_runtime_caches(self.root)

    def test_unverified_install_receipt_is_rejected(self) -> None:
        self.install_receipt("pending")
        with self.assertRaisesRegex(RuntimeError, "verified installed-package receipt"):
            MODULE.validate_runtime_caches(self.root)


if __name__ == "__main__":
    unittest.main()

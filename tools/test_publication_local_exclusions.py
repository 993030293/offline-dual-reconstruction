"""Regress local-only publication exclusions shared by build and verification."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load_module("publication_manifest_builder", HERE / "build_publication_manifest.py")
VERIFY = load_module("publication_verifier", HERE / "verify_publication.py")


class LocalExclusionTests(unittest.TestCase):
    def test_generated_json_uses_explicit_crlf_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            BUILD.write_json(path, {"value": "line"})
            payload = path.read_bytes()
            self.assertIn(b"\r\n", payload)
            self.assertNotIn(b"\n", payload.replace(b"\r\n", b""))
            self.assertTrue(payload.endswith(b"\r\n"))

    def test_machine_binding_name_is_local_only_at_any_depth(self):
        relative = "payload/02_code/reconstruction/OFFLINE_RUNTIME.local.json"
        self.assertTrue(BUILD.is_local_only_path(relative))
        self.assertTrue(VERIFY.is_local_only_path(relative))

    def test_machine_binding_is_absent_from_both_file_enumerators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "payload/02_code/reconstruction/OFFLINE_RUNTIME.local.json"
            local.parent.mkdir(parents=True)
            local.write_text("{}\n", encoding="utf-8")
            retained = root / "README.md"
            retained.write_text("retained\n", encoding="utf-8")
            built = {
                path.relative_to(root).as_posix()
                for path in BUILD.collect_files(root)
            }
            verified = VERIFY.actual_files(root)
            self.assertEqual(built, {"README.md"})
            self.assertEqual(verified, {"README.md"})


if __name__ == "__main__":
    unittest.main()

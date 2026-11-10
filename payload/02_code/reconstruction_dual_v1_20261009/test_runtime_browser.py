"""Verify explicit offline browser bindings without relying on machine-local files."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import importable_path
from native_quality_runtime import verify_native_quality_runtime
from package_delivery import browser_dependencies
from preflight import decode_process_output
from run_initial_safe import load_native
from run_supported_sparse import bind_reference_camera_tools


class RuntimeBrowserTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        root = Path(self.scratch.name)
        self.node = root / "node.exe"
        self.package = root / "playwright"
        self.node.write_bytes(b"node")
        self.package.mkdir()
        (self.package / "index.js").write_text("module.exports = {};\n", encoding="utf-8")

    def test_complete_explicit_binding_does_not_require_local_json(self):
        environment = {
            "RECON_NODE_EXECUTABLE": str(self.node),
            "RECON_PLAYWRIGHT_PACKAGE": str(self.package),
        }
        with patch.dict(os.environ, environment, clear=True):
            node, package = browser_dependencies()
        self.assertEqual(node, self.node)
        self.assertEqual(package, self.package)

    def test_partial_explicit_binding_fails_closed(self):
        with patch.dict(os.environ, {"RECON_NODE_EXECUTABLE": str(self.node)}, clear=True):
            with self.assertRaises(ValueError):
                browser_dependencies()

    def test_missing_explicit_paths_fail_closed(self):
        environment = {
            "RECON_NODE_EXECUTABLE": str(self.node.with_name("missing-node.exe")),
            "RECON_PLAYWRIGHT_PACKAGE": str(self.package),
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(FileNotFoundError):
                browser_dependencies()


class ProcessOutputDecodeTests(unittest.TestCase):
    def test_utf8_output_is_preserved(self):
        self.assertEqual(decode_process_output("GPU ready\n".encode("utf-8")), "GPU ready\n")

    def test_utf16_little_endian_output_without_bom_is_detected(self):
        encoded = "WSL warning\r\n".encode("utf-16-le")
        self.assertEqual(decode_process_output(encoded), "WSL warning\r\n")

    def test_utf16_output_with_bom_is_detected(self):
        encoded = "diagnostic\n".encode("utf-16")
        self.assertEqual(decode_process_output(encoded), "diagnostic\n")


class LongPathImportTests(unittest.TestCase):
    def test_absolute_windows_path_uses_extended_prefix(self):
        path = Path("C:/") / ("segment" * 40) / "module.py"
        value = importable_path(path)
        if os.name == "nt":
            self.assertTrue(value.startswith("\\\\?\\"))
        else:
            self.assertEqual(value, str(path))

    def test_installed_initial_frontend_imports_from_long_package_path(self):
        native = load_native()
        self.assertEqual(Path(native.decode_bayer.__code__.co_filename).name, "camera_motion.py")

    def test_sparse_augmentation_imports_camera_tools_from_long_package_path(self):
        camera_motion = bind_reference_camera_tools()
        self.assertEqual(Path(camera_motion.decode_bayer.__code__.co_filename).name, "camera_motion.py")


class NativeQualityRuntimeTests(unittest.TestCase):
    def test_bundled_libraries_match_accepted_component_results(self):
        native = load_native()
        receipt = verify_native_quality_runtime(native.demosaic_rggb8_to_bgr)
        self.assertEqual(receipt["status"], "bundled_native_quality_runtime_verified")
        self.assertEqual(receipt["bayer_exact_cases"], 18)
        self.assertEqual(receipt["planar_exact_cases"], 15)
        self.assertEqual(receipt["vector_exact_cases"], 18)


if __name__ == "__main__":
    unittest.main()

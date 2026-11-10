"""Verify the bundled native image-quality runtime against accepted CPU results."""

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

from common import ROOT


CODE = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
RUNTIME = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
LIBRARIES = {
    "bayer_gray": {
        "path": RUNTIME / "bg001/b0/native_bayer_gray.dll",
        "sha256": "dc740ba99da5084cdb51a1b990f653078e4b8a019640fbe422d9454868ea1787",
        "bytes": 104960,
    },
    "planar_corner": {
        "path": RUNTIME / "qc001/b0/native_planar_corner.dll",
        "sha256": "343359257441df65c16d26d871b1a0fc7b8c0acb3726b5cc4430a4dfe2153c16",
        "bytes": 126464,
    },
    "vector_gray_stats": {
        "path": RUNTIME / "vector_gray_stats_001/b0/native_vector_gray_stats.dll",
        "sha256": "f6bd5de17dfe58e700a6ada3dba048a6120dbe967c8b61970ca0eef60558d620",
        "bytes": 104960,
    },
}


def file_identity(path):
    """Return the byte identity used to freeze a bundled runtime library."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def arrays(value):
    """Normalize the OpenCV no-corner result for exact array comparisons."""
    return np.empty((0, 1, 2), np.float32) if value is None else value


def float_bits(values):
    """Expose exact binary64 results without tolerance-based comparisons."""
    return np.asarray(values, dtype=np.float64).view(np.uint64).tolist()


def verify_native_quality_runtime(demosaic_rggb8_to_bgr):
    """Load all bundled DLLs and verify exact deterministic component behavior."""
    identities = {}
    for name, expected in LIBRARIES.items():
        path = expected["path"]
        if not path.is_file():
            raise FileNotFoundError("A bundled native quality library is absent: " + str(path))
        actual = file_identity(path)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError("A bundled native quality library identity changed: " + name)
        identities[name] = actual

    code = str(CODE)
    if code not in sys.path:
        sys.path.insert(0, code)
    from exact_bayer_gray import BayerGrayEngine, BayerGrayWorkspace
    from planar_corner import ORIGINAL_CORNER, PARAMETERS, PlanarCornerWorkspace
    from quality_histogram import gray_moments
    from vector_gray_stats import VectorGrayStats

    old_threads = cv2.getNumThreads()
    bayer_engine = BayerGrayEngine(LIBRARIES["bayer_gray"]["path"])
    bayer_workspace = BayerGrayWorkspace(bayer_engine)
    planar_workspace = PlanarCornerWorkspace(LIBRARIES["planar_corner"]["path"])
    vector_engine = VectorGrayStats(LIBRARIES["vector_gray_stats"]["path"])
    checks = {
        "opencv_version": cv2.__version__,
        "opencv_optimized": bool(cv2.useOptimized()),
        "opencv_ipp": bool(cv2.ipp.useIPP()),
        "libraries": identities,
    }
    rng = np.random.default_rng(20261110)
    try:
        cv2.setNumThreads(1)
        if cv2.__version__ != "4.13.0" or not cv2.useOptimized() or not cv2.ipp.useIPP():
            raise RuntimeError("The accepted OpenCV native-quality execution contract changed")
        if not all((bayer_engine.available, planar_workspace.available, vector_engine.available)):
            raise RuntimeError("The current CPU cannot execute every bundled AVX2 quality library")

        bayer_cases = []
        for shape in ((1, 1), (2, 23), (23, 2), (37, 49), (64, 65), (620, 812)):
            y, x = np.indices(shape)
            values = (
                rng.integers(0, 256, shape, dtype=np.uint8),
                ((x * 3 + y * 5) % 256).astype(np.uint8),
                (((x // 4 + y // 4) % 2) * 255).astype(np.uint8),
            )
            for raw in values:
                immutable = raw.copy()
                actual = bayer_workspace.gray(raw).copy()
                expected = cv2.cvtColor(demosaic_rggb8_to_bgr(raw), cv2.COLOR_BGR2GRAY)
                if not np.array_equal(actual, expected) or not np.array_equal(raw, immutable):
                    raise ValueError("The bundled Bayer gray DLL changed an accepted pixel result")
                bayer_cases.append({"shape": list(shape), "pixels": int(raw.size)})

        planar_cases = []
        for shape in ((1, 1), (2, 7), (31, 37), (120, 160), (620, 812)):
            y, x = np.indices(shape)
            values = (
                np.zeros(shape, np.uint8),
                ((x * 3 + y * 5) % 256).astype(np.uint8),
                rng.integers(0, 256, shape, dtype=np.uint8),
            )
            for gray in values:
                expected_response = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
                actual_response = planar_workspace.response(gray).copy()
                expected_corners = ORIGINAL_CORNER(gray, **PARAMETERS)
                actual_corners = planar_workspace.goodFeaturesToTrack(gray, **PARAMETERS)
                if not np.array_equal(expected_response.view(np.uint32), actual_response.view(np.uint32)):
                    raise ValueError("The bundled planar corner DLL changed response bits")
                if ((expected_corners is None) != (actual_corners is None)
                        or not np.array_equal(arrays(expected_corners), arrays(actual_corners))):
                    raise ValueError("The bundled planar corner DLL changed coordinates or order")
                planar_cases.append({"shape": list(shape), "corners": len(arrays(actual_corners))})

        vector_cases = []
        for shape in ((1, 1), (1, 31), (31, 1), (3, 17), (33, 31), (620, 812)):
            y, x = np.indices(shape)
            values = (
                rng.integers(0, 256, shape, dtype=np.uint8),
                (((x + y) % 2) * 255).astype(np.uint8),
                np.full(shape, 250, np.uint8),
            )
            for gray in values:
                derivative = cv2.Laplacian(gray, cv2.CV_16S).astype(np.int64)
                expected_integers = (
                    int(gray.sum(dtype=np.uint64)),
                    int(np.count_nonzero((gray <= 5) | (gray >= 250))),
                    int(derivative.sum(dtype=np.int64)),
                    int((derivative * derivative).sum(dtype=np.int64)),
                )
                actual_integers = vector_engine.moments(gray)
                _, deviation = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
                brightness, clipped = gray_moments(gray)
                expected_quality = float(deviation[0, 0] ** 2), brightness, clipped
                actual_quality = vector_engine.quality_statistics(gray)
                if actual_integers != expected_integers:
                    raise ValueError("The bundled vector statistics DLL changed integer accumulators")
                if float_bits(actual_quality) != float_bits(expected_quality):
                    raise ValueError("The bundled vector statistics DLL changed binary64 quality values")
                vector_cases.append({"shape": list(shape), "float_bits": float_bits(actual_quality)})

        checks.update(
            status="bundled_native_quality_runtime_verified",
            bayer_exact_cases=len(bayer_cases),
            planar_exact_cases=len(planar_cases),
            vector_exact_cases=len(vector_cases),
            bayer_cases=bayer_cases,
            planar_cases=planar_cases,
            vector_cases=vector_cases,
            original_cv2_corner_unchanged=cv2.goodFeaturesToTrack is ORIGINAL_CORNER,
        )
        return checks
    finally:
        vector_engine.close()
        planar_workspace.close()
        bayer_workspace.close()
        bayer_engine.close()
        cv2.setNumThreads(old_threads)


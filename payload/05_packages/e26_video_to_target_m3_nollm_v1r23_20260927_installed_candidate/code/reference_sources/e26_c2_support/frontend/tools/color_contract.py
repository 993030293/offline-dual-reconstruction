"""Shared Bayer and target-colour contracts for the reconstruction tools."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


COLOR_CONTRACT_VERSION = "rggb8_bgr_paperbox_ochre_v2"
PAPERBOX_HSV_LOWER = np.asarray([6, 30, 25], dtype=np.uint8)
PAPERBOX_HSV_UPPER = np.asarray([22, 255, 220], dtype=np.uint8)


def demosaic_rggb8_to_bgr(raw: np.ndarray) -> np.ndarray:
    """Decode a declared ``bayer_rggb8`` frame into OpenCV BGR order."""
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("raw Bayer frame must be a two-dimensional uint8 array")
    # Use the explicit four-letter pattern. OpenCV's legacy BayerRG alias maps
    # to a different conversion code and swaps red and blue for RGGB input.
    return cv2.cvtColor(raw, cv2.COLOR_BayerRGGB2BGR)


def paperbox_color_mask_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Return the measured ochre/yellow-brown target-colour mask in BGR input."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 BGR array")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, PAPERBOX_HSV_LOWER, PAPERBOX_HSV_UPPER)


def require_target_color_fraction(record: Mapping[str, object]) -> float:
    """Read the corrected selector field and reject stale blue-channel scouts."""
    if "target_color_fraction" not in record:
        raise ValueError(
            "scout record lacks target_color_fraction; regenerate it with the "
            f"{COLOR_CONTRACT_VERSION} color contract"
        )
    return float(record["target_color_fraction"])


def require_color_contract(record: Mapping[str, object]) -> None:
    """Reject derived records made with another channel or colour contract."""
    actual = record.get("color_contract")
    if actual != COLOR_CONTRACT_VERSION:
        raise ValueError(
            f"expected color_contract={COLOR_CONTRACT_VERSION}, got {actual!r}; "
            "regenerate the derived record"
        )

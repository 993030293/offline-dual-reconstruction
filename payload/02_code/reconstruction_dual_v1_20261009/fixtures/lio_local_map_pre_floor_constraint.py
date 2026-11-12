"""Retained pre-floor-constraint ICP reference used only for regression tests.

This function was copied from the package's pre-repair source snapshot. It is
not imported by the runtime pipeline and has no file, network, or model access.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def point_to_plane_translation_icp(
    source: np.ndarray,
    target: np.ndarray,
    rotation: Rotation,
    initial_translation: np.ndarray,
    maximum_correspondence: float,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    rotated = rotation.apply(source)
    translation = initial_translation.copy()
    tree = cKDTree(target)
    previous_plane_rmse = np.inf
    metrics = {
        "correspondences": 0,
        "rmse_m": np.inf,
        "plane_rmse_m": np.inf,
        "iterations": 0,
    }
    for iteration in range(iterations):
        moved = rotated + translation
        distances, neighbors = tree.query(moved, k=1, workers=-1)
        eligible = np.flatnonzero(distances < maximum_correspondence)
        if eligible.size < 50:
            break
        trim_limit = np.percentile(distances[eligible], 75)
        selected = eligible[distances[eligible] <= trim_limit]
        matched = target[neighbors[selected]]
        _, normal_neighbors = tree.query(matched, k=12, workers=-1)
        neighborhoods = target[normal_neighbors]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / 11.0
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        residual = np.einsum("ij,ij->i", normals, moved[selected] - matched)
        delta, _, _, _ = np.linalg.lstsq(normals, -residual, rcond=None)
        if np.linalg.norm(delta) > 0.2:
            delta *= 0.2 / np.linalg.norm(delta)
        translation += delta
        point_rmse = float(np.sqrt(np.mean(distances[selected] ** 2)))
        plane_rmse = float(np.sqrt(np.mean(residual**2)))
        metrics = {
            "correspondences": int(selected.size),
            "rmse_m": point_rmse,
            "plane_rmse_m": plane_rmse,
            "iterations": iteration + 1,
        }
        if (
            abs(previous_plane_rmse - plane_rmse) < 1e-5
            and np.linalg.norm(delta) < 1e-4
        ):
            break
        previous_plane_rmse = plane_rmse
    return translation, metrics

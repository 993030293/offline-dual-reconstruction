"""Preserve observed non-ground geometry when tightening a validated target ROI."""

import numpy as np

POLICY_VERSION = "observed_non_ground_surface_preservation_v1"


def protected_triangles(vertices, triangles, normal, offset, ground_policy):
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    normal = np.asarray(normal, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or normal.shape != (3,) or not np.isfinite(vertices).all()
            or not np.isfinite(normal).all() or not np.isfinite(offset)
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6)
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("Malformed observed surface or metric floor")
    maximum_ground = float(ground_policy["maximum_signed_height_m"])
    minimum_cosine = float(ground_policy["minimum_normal_cosine"])
    unconditional_low = float(ground_policy["unconditional_low_artifact_maximum_height_m"])
    if (not np.isfinite([maximum_ground, minimum_cosine, unconditional_low]).all()
            or not 0 <= minimum_cosine <= 1 or unconditional_low > maximum_ground):
        raise ValueError("Invalid existing ground classification policy")
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    height = (vertices @ normal + offset)[triangles].max(axis=1)
    cosine = np.abs(cross @ normal) / np.maximum(2 * area, 1e-20)
    # Protect elevated and contact-side observations using the existing ground policy.
    protected = (height > maximum_ground) | ((cosine < minimum_cosine) & (height > unconditional_low))
    return protected, area


def choose_preserving_candidate(rows, maximum_support_drop):
    if not np.isfinite(maximum_support_drop) or maximum_support_drop < 0:
        raise ValueError("Invalid sparse support loss bound")
    for row in rows:
        if row.get("status") == "empty":
            continue
        lost, drop = row["protected_triangles_lost"], row["absolute_support_drop"]
        if not np.isfinite(drop) or drop < 0 or not isinstance(lost, int) or lost < 0:
            raise ValueError("Invalid observed-surface preservation evidence")
        if lost == 0 and drop <= maximum_support_drop:
            return row
    raise ValueError("No ROI candidate preserved observed non-ground geometry and sparse support")


def trim_observed_surface(mesh, support, roi, ground_policy, reference):
    import open3d as o3d
    from scipy.spatial import cKDTree

    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    floor = roi["roi_derivation"]["spatial_sparse_floor_plane"]
    protected, area = protected_triangles(vertices, triangles, floor["normal"], floor["offset_m"], ground_policy)
    support = np.asarray(support, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 3 or not len(support) or not np.isfinite(support).all():
        raise ValueError("Invalid metric sparse support")
    lower, upper = (np.asarray(roi["roi_world_lio_m"][key], dtype=float) for key in ("minimum", "maximum"))
    quantiles = roi["roi_derivation"]["metric_sparse_vehicle_support"]
    qlower = np.asarray(quantiles["quantile_minimum_xyz_m"], dtype=float)
    qupper = np.asarray(quantiles["quantile_maximum_xyz_m"], dtype=float)
    if (any(value.shape != (3,) or not np.isfinite(value).all() for value in (lower, upper, qlower, qupper))
            or not (upper > lower).all() or not (qupper[:2] > qlower[:2]).all()
            or (vertices < lower - 1e-6).any() or (vertices > upper + 1e-6).any()):
        raise ValueError("Observed surface or quantile bounds exceed the validated source ROI")
    paddings = list(reference.ROI_TRIM_PADDING_CANDIDATES_M)
    if not paddings or not np.isfinite(paddings).all() or any(p < 0 for p in paddings) or sorted(set(paddings)) != paddings:
        raise ValueError("Invalid original ROI candidate paddings")
    baseline = float(np.mean(cKDTree(vertices).query(support, workers=1)[0] <= reference.SPARSE_SUPPORT_DISTANCE_M))
    rows, chosen = [], None
    for padding in paddings:
        lo, hi = lower.copy(), upper.copy()
        lo[:2], hi[:2] = qlower[:2] - padding, qupper[:2] + padding
        if (lo[:2] < lower[:2] - 1e-6).any() or (hi[:2] > upper[:2] + 1e-6).any():
            raise ValueError("ROI tightening may not expand beyond validated source bounds")
        if padding == paddings[-1]:
            candidate, kept, lo, hi = mesh, np.ones(len(triangles), dtype=bool), lower.copy(), upper.copy()
        else:
            inside = ((vertices >= lo) & (vertices <= hi)).all(axis=1)
            kept = inside[triangles].all(axis=1)
            candidate = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        if not candidate.has_triangles():
            rows.append({"padding_m": padding, "status": "empty"})
            continue
        if len(candidate.triangles) != int(kept.sum()):
            raise ValueError("Native ROI crop semantics differ from the preservation test")
        fraction = float(np.mean(cKDTree(np.asarray(candidate.vertices)).query(support, workers=1)[0]
                                 <= reference.SPARSE_SUPPORT_DISTANCE_M))
        lost = protected & ~kept
        row = {"padding_m": padding, "triangle_count": len(candidate.triangles),
               "supported_fraction_within_5cm": fraction, "absolute_support_drop": max(0., baseline - fraction),
               "protected_triangles_lost": int(lost.sum()), "protected_area_lost_m2": float(area[lost].sum())}
        row["status"] = ("observed_surface_loss" if row["protected_triangles_lost"] else
                         "support_loss" if row["absolute_support_drop"] > reference.MAXIMUM_ROI_TRIM_SUPPORT_DROP else "accepted")
        rows.append(row)
        if row["status"] == "accepted":
            chosen = (candidate, lo, hi)
            break
    selected = choose_preserving_candidate(rows, reference.MAXIMUM_ROI_TRIM_SUPPORT_DROP)
    if chosen is None:
        raise ValueError("No observed-surface-preserving crop was produced")
    candidate, lo, hi = chosen
    evidence = {"method": "smallest sparse-supported ROI preserving observed non-ground triangles",
                "policy_version": POLICY_VERSION, "candidate_padding_m": paddings,
                "maximum_absolute_sparse_support_drop": reference.MAXIMUM_ROI_TRIM_SUPPORT_DROP,
                "reference_support_point_count": len(support), "source_supported_fraction_within_5cm": baseline,
                "selected_padding_m": selected["padding_m"], "source_triangle_count": len(triangles),
                "selected_triangle_count": len(candidate.triangles), "protected_triangle_count": int(protected.sum()),
                "protected_triangles_lost": selected["protected_triangles_lost"],
                "ground_classification_policy": dict(ground_policy), "new_triangle_geometry_created": False,
                "effective_roi_world_lio_m": {"minimum": lo.tolist(), "maximum": hi.tolist()}, "candidates": rows}
    return candidate, evidence

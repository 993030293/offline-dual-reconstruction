#!/usr/bin/env python
"""Build the D1 one-million-point offline diagnostic viewer."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go


V2_PATH = Path(__file__).with_name("build_pointcloud_interactive_v2.py")
QUALITY_LABELS = ("performance", "balanced", "detail", "ultra")
DEFAULT_QUALITY_TARGETS = (150_000, 350_000, 650_000, 1_000_000)
AUTO_POINT_SIZES = {
    "performance": 1.70,
    "balanced": 1.35,
    "detail": 1.15,
    "ultra": 1.00,
}
DEFAULT_OPACITY = 0.90
DEFAULT_LAYER_STATE = "s08_terminal_ground_removed_before_component_filter"
DEFAULT_BOUNDARY_HEADING = "S08 终点"
DEFAULT_SUBTITLE = "目标 ROI 观测表面点云预览 · 只读显示层 · 无补点 / 无补面"
DEFAULT_BOUNDARY_NOTE = (
    "当前显示去除地面与无条件低位伪影后的点云。"
    "S09 连通组件过滤及其后续阶段均未运行。"
)
DEFAULT_CLAIM_BOUNDARY = (
    "S08 display-only engineering preview. "
    "S09 connected-component filtering and every later stage are excluded."
)


def _load_v2_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("pointcloud_viewer_v2", V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the v2 viewer dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_v2_module()
sha256 = V2.sha256
read_xyzrgb_ply = V2.read_xyzrgb_ply
progressive_spatial_indices = V2.progressive_spatial_indices


def validate_quality_targets(values: Sequence[int]) -> tuple[int, int, int, int]:
    if len(values) != len(QUALITY_LABELS):
        raise ValueError("Exactly four quality point targets are required")
    targets = tuple(int(value) for value in values)
    if targets[0] <= 0 or any(left >= right for left, right in zip(targets, targets[1:])):
        raise ValueError("Quality point targets must be positive and strictly increasing")
    return targets  # type: ignore[return-value]


def _rgb_strings(colors: np.ndarray) -> list[str]:
    return [f"rgb({red},{green},{blue})" for red, green, blue in colors]


def _horizontal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = float(points[:, 2].min())
    upper = float(points[:, 2].max())
    subset = points[points[:, 2] <= lower + 0.58 * max(upper - lower, 1e-6)]
    if subset.shape[0] < 16:
        subset = points
    stride = max(1, subset.shape[0] // 300_000)
    xy = subset[::stride, :2].astype(np.float64)
    center = np.median(xy, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((xy - center).T))
    primary = eigenvectors[:, int(np.argmax(eigenvalues))]
    if primary[0] < 0:
        primary = -primary
    primary /= np.linalg.norm(primary)
    secondary = np.array([-primary[1], primary[0]], dtype=np.float64)
    if secondary[1] < 0:
        secondary = -secondary
    return primary, secondary


def _rotate_xy(vector: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [
            math.cos(angle) * vector[0] - math.sin(angle) * vector[1],
            math.sin(angle) * vector[0] + math.cos(angle) * vector[1],
        ],
        dtype=np.float64,
    )


def subject_camera_rows(points: np.ndarray) -> dict[str, dict[str, Any]]:
    primary, secondary = _horizontal_axes(points)
    candidates = [primary, -primary, secondary, -secondary]
    visible_direction = np.array([1.0, -1.0], dtype=np.float64)
    visible_direction /= np.linalg.norm(visible_direction)
    candidates.sort(
        key=lambda vector: float(np.dot(vector, visible_direction)), reverse=True
    )
    face_a = candidates[0]
    face_b = candidates[1]

    def row(vector: np.ndarray, z: float = 0.35) -> dict[str, Any]:
        return {
            "eye": {
                "x": float(2.35 * vector[0]),
                "y": float(2.35 * vector[1]),
                "z": z,
            },
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        }

    return {
        "subjectFaceA": row(face_a),
        "subjectFaceB": row(face_b),
        "rightGrazingMinus": row(_rotate_xy(face_a, -25.0)),
        "rightGrazingPlus": row(_rotate_xy(face_a, 25.0)),
    }


def spatial_coverage(
    points: np.ndarray,
    progressive_indices: np.ndarray,
    actual_counts: dict[str, int],
    *,
    grid_levels: Sequence[int] = (64, 128, 256),
) -> dict[str, dict[str, float]]:
    minimum = points.min(axis=0).astype(np.float64)
    span = points.max(axis=0).astype(np.float64) - minimum
    span[span == 0] = 1.0
    normalized = np.clip(
        (points.astype(np.float64) - minimum) / span,
        0.0,
        np.nextafter(1.0, 0.0),
    )
    result: dict[str, dict[str, float]] = {label: {} for label in QUALITY_LABELS}
    for level in grid_levels:
        cells = np.floor(normalized * level).astype(np.int64)
        codes = cells[:, 0] + level * (cells[:, 1] + level * cells[:, 2])
        full_count = int(np.unique(codes).shape[0])
        for label in QUALITY_LABELS:
            count = actual_counts[label]
            occupied = int(np.unique(codes[progressive_indices[:count]]).shape[0])
            result[label][str(level)] = round(occupied / max(full_count, 1), 6)
    return result


def build_figure(
    points: np.ndarray,
    colors: np.ndarray,
    progressive_indices: np.ndarray,
    quality_targets: tuple[int, int, int, int],
) -> tuple[go.Figure, dict[str, int]]:
    actual_counts = {
        label: min(int(points.shape[0]), target)
        for label, target in zip(QUALITY_LABELS, quality_targets, strict=True)
    }
    boundaries = (0, *actual_counts.values())
    traces: list[go.Scatter3d] = []
    for layer_index, label in enumerate(QUALITY_LABELS):
        start = boundaries[layer_index]
        stop = boundaries[layer_index + 1]
        layer_indices = progressive_indices[start:stop]
        layer_points = points[layer_indices]
        traces.append(
            go.Scatter3d(
                x=layer_points[:, 0],
                y=layer_points[:, 1],
                z=layer_points[:, 2],
                mode="markers",
                marker={
                    "size": AUTO_POINT_SIZES["balanced"],
                    "color": _rgb_strings(colors[layer_indices]),
                    "opacity": DEFAULT_OPACITY,
                },
                hoverinfo="skip",
                name=label,
                visible=layer_index < 2,
            )
        )
    lower = points.min(axis=0).astype(np.float64)
    upper = points.max(axis=0).astype(np.float64)
    margin = 0.025 * np.maximum(upper - lower, 1e-6)
    axis_style = {
        "showbackground": True,
        "backgroundcolor": "#080b0f",
        "gridcolor": "#273849",
        "zerolinecolor": "#52677c",
        "tickfont": {"color": "#8fa2b5", "size": 11},
    }
    axis_title_font = {"color": "#aebdca", "size": 12}
    figure = go.Figure(data=traces)
    figure.update_layout(
        paper_bgcolor="#080b0f",
        plot_bgcolor="#080b0f",
        font={"color": "#d9e2ea", "family": "Segoe UI, sans-serif"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        uirevision="e26-pointcloud-inspector-v3",
        scene={
            "aspectmode": "data",
            "bgcolor": "#080b0f",
            "camera": {
                "eye": {"x": 1.55, "y": -1.55, "z": 1.05},
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "projection": {"type": "orthographic"},
            },
            "xaxis": {
                **axis_style,
                "title": {"text": "x (m)", "font": axis_title_font},
                "range": [float(lower[0] - margin[0]), float(upper[0] + margin[0])],
            },
            "yaxis": {
                **axis_style,
                "title": {"text": "y (m)", "font": axis_title_font},
                "range": [float(lower[1] - margin[1]), float(upper[1] + margin[1])],
            },
            "zaxis": {
                **axis_style,
                "title": {"text": "z (m)", "font": axis_title_font},
                "range": [float(lower[2] - margin[2]), float(upper[2] + margin[2])],
            },
        },
    )
    return figure, actual_counts


def _replace_once(document: str, old: str, new: str) -> str:
    if document.count(old) != 1:
        raise RuntimeError(f"Expected exactly one v2 document token: {old[:80]!r}")
    return document.replace(old, new, 1)


def build_document(
    figure: go.Figure,
    *,
    title: str,
    source_point_count: int,
    actual_counts: dict[str, int],
    coverage: dict[str, dict[str, float]],
    camera_rows: dict[str, dict[str, Any]],
    layer_state: str = DEFAULT_LAYER_STATE,
    subtitle: str = DEFAULT_SUBTITLE,
    boundary_heading: str = DEFAULT_BOUNDARY_HEADING,
    boundary_note: str = DEFAULT_BOUNDARY_NOTE,
) -> str:
    document = V2.build_document(
        figure,
        title=title,
        source_point_count=source_point_count,
        actual_counts=actual_counts,
    )
    document = _replace_once(
        document,
        "重力校准裁切点云 · 只读显示层 · 无补点 / 无补面 / 无 Mesh",
        html.escape(subtitle),
    )
    base_meta = json.dumps(
        {"source": int(source_point_count), "counts": actual_counts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    extended_meta = json.dumps(
        {
            "source": int(source_point_count),
            "counts": actual_counts,
            "coverage": coverage,
            "layerState": layer_state,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    document = _replace_once(document, f"const meta = {base_meta};", f"const meta = {extended_meta};")
    document = _replace_once(
        document,
        '<button data-quality="detail">精细</button>',
        '<button data-quality="detail">精细</button>\n            <button data-quality="ultra">超清</button>',
    )
    document = _replace_once(
        document,
        ".button-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }",
        ".button-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }",
    )
    balanced_point_size = f"{AUTO_POINT_SIZES['balanced']:.2f}"
    document = _replace_once(
        document,
        '<span class="value" id="point-size-value">1.00 px</span>',
        f'<span class="value" id="point-size-value">{balanced_point_size} px</span>',
    )
    document = _replace_once(
        document,
        '<input id="point-size" type="range" min="0.25" max="3.00" step="0.05" value="1.00" />',
        f'<input id="point-size" type="range" min="0.25" max="3.00" step="0.05" value="{balanced_point_size}" />\n'
        '          <div class="toggle-row" style="margin-top:8px">\n'
        '            <button class="active" id="point-size-auto">点径：自动</button>\n'
        '            <button class="active" id="projection-toggle">投影：正交</button>\n'
        '          </div>',
    )
    document = _replace_once(
        document,
        '<button data-camera="left">左侧</button><button data-camera="top">顶视</button><button class="active" data-camera="oblique">斜视</button>',
        '<button data-camera="left">左侧</button><button data-camera="top">顶视</button><button class="active" data-camera="oblique">斜视</button>\n'
        '            <button data-camera="subjectFaceA">主体面 A</button><button data-camera="subjectFaceB">主体面 B</button>\n'
        '            <button data-camera="rightGrazingMinus">A 掠射−</button><button data-camera="rightGrazingPlus">A 掠射＋</button>',
    )
    document = _replace_once(
        document,
        '<div class="status" id="status"></div>',
        '<div class="status" id="status"></div>\n'
        f'        <div class="status"><strong>{html.escape(boundary_heading)}</strong><br>'
        f'{html.escape(boundary_note)}</div>',
    )
    document = _replace_once(
        document,
        '数字 1/2/3 切换密度',
        '数字 1/2/3/4 切换密度',
    )
    document = _replace_once(
        document,
        'const qualityNames = { performance: "流畅", balanced: "平衡", detail: "精细" };',
        'const qualityNames = { performance: "流畅", balanced: "平衡", detail: "精细", ultra: "超清" };',
    )
    document = _replace_once(
        document,
        'const visibility = { performance: [true, false, false], balanced: [true, true, false], detail: [true, true, true] };',
        'const visibility = { performance: [true, false, false, false], balanced: [true, true, false, false], detail: [true, true, true, false], ultra: [true, true, true, true] };',
    )
    oblique_line = 'oblique: { eye: {x: 1.55, y: -1.55, z: 1.05}, up: {x:0,y:0,z:1}, center: {x:0,y:0,z:0} }'
    camera_lines = [oblique_line]
    for name, row in camera_rows.items():
        camera_lines.append(f"      {name}: {json.dumps(row, separators=(',', ':'))}")
    document = _replace_once(document, oblique_line, ",\n".join(camera_lines))
    document = _replace_once(
        document,
        'let light = false;',
        'let light = false;\n    let autoPointSize = true;\n    let orthographic = true;\n'
        f'    const autoPointSizes = {json.dumps(AUTO_POINT_SIZES, separators=(",", ":"))};',
    )
    document = _replace_once(
        document,
        'status.innerHTML = `<strong>${qualityNames[quality]}</strong> · 显示 ${countText(meta.counts[quality])} / ${countText(meta.source)} 点<br><span class="badge">空间分层抽样</span>仅改变预览密度，源 PLY 不变。`;',
        'status.innerHTML = `<strong>${qualityNames[quality]}</strong> · 显示 ${countText(meta.counts[quality])} / ${countText(meta.source)} 点<br><span class="badge">空间分层抽样</span> 128³ 占据覆盖 ${Number(meta.coverage[quality]["128"] * 100).toFixed(2)}% · 源 PLY 不变。`;',
    )
    document = _replace_once(
        document,
        'Plotly.restyle(graph, {visible: visibility[next]}, [0, 1, 2]);',
        'Plotly.restyle(graph, {visible: visibility[next]}, [0, 1, 2, 3]);',
    )
    document = _replace_once(
        document,
        'updateStatus();\n    }\n    function setPointSize(value)',
        'updateStatus();\n      if (autoPointSize) setPointSize(autoPointSizes[next]);\n    }\n    function setPointSize(value)',
    )
    document = _replace_once(
        document,
        'Plotly.restyle(graph, {"marker.size": parsed}, [0, 1, 2]);',
        'Plotly.restyle(graph, {"marker.size": parsed}, [0, 1, 2, 3]);',
    )
    document = _replace_once(
        document,
        'Plotly.restyle(graph, {"marker.opacity": parsed}, [0, 1, 2]);',
        'Plotly.restyle(graph, {"marker.opacity": parsed}, [0, 1, 2, 3]);',
    )
    document = _replace_once(
        document,
        'document.getElementById("point-size").addEventListener("input", event => setPointSize(event.target.value));',
        'function setManualPointSize(value) { autoPointSize = false; document.getElementById("point-size-auto").classList.remove("active"); document.getElementById("point-size-auto").textContent = "点径：手动"; setPointSize(value); }\n'
        '    document.getElementById("point-size").addEventListener("input", event => setManualPointSize(event.target.value));\n'
        '    document.getElementById("point-size-auto").addEventListener("click", event => { autoPointSize = !autoPointSize; event.target.classList.toggle("active", autoPointSize); event.target.textContent = autoPointSize ? "点径：自动" : "点径：手动"; if (autoPointSize) setPointSize(autoPointSizes[quality]); });\n'
        '    document.getElementById("projection-toggle").addEventListener("click", event => { orthographic = !orthographic; Plotly.relayout(graph, {"scene.camera.projection.type": orthographic ? "orthographic" : "perspective"}); event.target.classList.toggle("active", orthographic); event.target.textContent = `投影：${orthographic ? "正交" : "透视"}`; });',
    )
    document = _replace_once(
        document,
        'Plotly.relayout(graph, {"scene.camera": cameras[name]});',
        'Plotly.relayout(graph, {"scene.camera": {...cameras[name], projection: {type: orthographic ? "orthographic" : "perspective"}}});',
    )
    document = _replace_once(
        document,
        'if (event.key === "3") setQuality("detail");',
        'if (event.key === "3") setQuality("detail");\n      if (event.key === "4") setQuality("ultra");',
    )
    document = _replace_once(
        document,
        'if (event.key === "[") setPointSize(Math.max(.25, Number(document.getElementById("point-size").value) - .1));',
        'if (event.key === "[") setManualPointSize(Math.max(.25, Number(document.getElementById("point-size").value) - .1));',
    )
    document = _replace_once(
        document,
        'if (event.key === "]") setPointSize(Math.min(3, Number(document.getElementById("point-size").value) + .1));',
        'if (event.key === "]") setManualPointSize(Math.min(3, Number(document.getElementById("point-size").value) + .1));',
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--layer-state", default=DEFAULT_LAYER_STATE)
    parser.add_argument("--subtitle", default=DEFAULT_SUBTITLE)
    parser.add_argument("--boundary-heading", default=DEFAULT_BOUNDARY_HEADING)
    parser.add_argument("--boundary-note", default=DEFAULT_BOUNDARY_NOTE)
    parser.add_argument("--claim-boundary", default=DEFAULT_CLAIM_BOUNDARY)
    parser.add_argument(
        "--component-filter-state",
        choices=("not_applied", "applied_machine_geometry"),
        default="not_applied",
    )
    parser.add_argument(
        "--quality-points",
        nargs=4,
        type=int,
        default=DEFAULT_QUALITY_TARGETS,
        metavar=("PERFORMANCE", "BALANCED", "DETAIL", "ULTRA"),
    )
    args = parser.parse_args()
    quality_targets = validate_quality_targets(args.quality_points)
    if args.output_html.exists() or args.output_receipt.exists():
        raise FileExistsError("Output HTML and receipt must not already exist")

    input_hash_before = sha256(args.input_ply)
    points, colors = read_xyzrgb_ply(args.input_ply)
    maximum_points = min(int(points.shape[0]), quality_targets[-1])
    indices = progressive_spatial_indices(points, maximum_points)
    figure, actual_counts = build_figure(points, colors, indices, quality_targets)
    coverage = spatial_coverage(points, indices, actual_counts)
    cameras = subject_camera_rows(points)
    document = build_document(
        figure,
        title=args.title,
        source_point_count=int(points.shape[0]),
        actual_counts=actual_counts,
        coverage=coverage,
        camera_rows=cameras,
        layer_state=args.layer_state,
        subtitle=args.subtitle,
        boundary_heading=args.boundary_heading,
        boundary_note=args.boundary_note,
    )
    if re.search(r"<script\b[^>]*\bsrc\s*=", document, flags=re.IGNORECASE):
        raise RuntimeError("Viewer contains an external script dependency")
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    with args.output_html.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    input_hash_after = sha256(args.input_ply)
    if input_hash_after != input_hash_before:
        raise RuntimeError("Source PLY changed while building the viewer")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    receipt = {
        "schema_version": "e26_pointcloud_interactive_viewer_receipt_v3",
        "status": "complete_engineering_preview",
        "stage": "D1",
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__)),
        "v2_dependency_path": str(V2_PATH.resolve()),
        "v2_dependency_sha256": sha256(V2_PATH),
        "input_ply": str(args.input_ply.resolve()),
        "input_ply_sha256": input_hash_before,
        "input_point_count": int(points.shape[0]),
        "display_point_counts": actual_counts,
        "maximum_embedded_point_count": int(maximum_points),
        "sampling": {
            "algorithm": "deterministic_progressive_multiscale_voxel_strata_v1",
            "nested_quality_prefixes": True,
            "source_order_even_index_sampling": False,
        },
        "spatial_coverage": coverage,
        "viewer_defaults": {
            "quality": "balanced",
            "point_size_mode": "automatic",
            "auto_point_sizes_px": AUTO_POINT_SIZES,
            "opacity": DEFAULT_OPACITY,
            "projection": "orthographic",
            "camera": "oblique_z_up",
        },
        "exact_axis_lower_xyz_m": [float(value) for value in lower],
        "exact_axis_upper_xyz_m": [float(value) for value in upper],
        "axis_padding_fraction": 0.025,
        "viewer_controls": [
            "quality_150k_350k_650k_1000k",
            "manual_and_automatic_point_size",
            "point_opacity",
            "global_and_subject_aligned_camera_presets",
            "right_face_grazing_presets",
            "orthographic_perspective_toggle",
            "exact_source_bounds_with_padding",
            "grid_and_background_toggles",
        ],
        "layer_state": args.layer_state,
        "component_filter_state": args.component_filter_state,
        "subtitle": args.subtitle,
        "offline_self_contained": True,
        "external_script_count": 0,
        "output_html": str(args.output_html.resolve()),
        "output_html_bytes": args.output_html.stat().st_size,
        "output_html_sha256": sha256(args.output_html),
        "source_ply_unchanged": True,
        "source_ply_sha256_after": input_hash_after,
        "crop_modification_count": 0,
        "synthetic_point_generation_count": 0,
        "surface_completion_count": 0,
        "mesh_generation_count": 0,
        "claim_boundary": args.claim_boundary,
        "process_id": os.getpid(),
    }
    with args.output_receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

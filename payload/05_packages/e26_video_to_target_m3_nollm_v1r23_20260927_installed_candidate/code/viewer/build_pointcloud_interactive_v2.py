#!/usr/bin/env python
"""Build a self-contained, tunable RGB point-cloud inspection viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html


QUALITY_LABELS = ("performance", "balanced", "detail")
SAMPLING_LEVELS = (8, 16, 32, 64, 128, 256, 512)
DEFAULT_QUALITY_TARGETS = (150_000, 350_000, 650_000)
DEFAULT_POINT_SIZE = 1.0
DEFAULT_OPACITY = 0.9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_xyzrgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header is incomplete")
            line = raw.decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("Only binary little-endian PLY is supported")
        vertex_lines = [line for line in header_lines if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError("PLY must contain exactly one vertex declaration")
        count = int(vertex_lines[0].split()[-1])
        expected_properties = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        properties = [line for line in header_lines if line.startswith("property ")]
        if properties != expected_properties:
            raise ValueError(f"Unexpected PLY property schema: {properties}")
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.fromfile(stream, dtype=dtype, count=count)
        if vertices.shape[0] != count or stream.read(1):
            raise ValueError("PLY payload length does not match its header")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    return points, colors


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mixed = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        mixed += np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return mixed ^ (mixed >> np.uint64(31))


def progressive_spatial_indices(
    points: np.ndarray,
    maximum_points: int,
    *,
    grid_levels: Sequence[int] = SAMPLING_LEVELS,
) -> np.ndarray:
    """Return a deterministic coarse-to-fine spatially distributed point order."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if maximum_points <= 0:
        raise ValueError("maximum_points must be positive")
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")
    if any(level <= 1 for level in grid_levels):
        raise ValueError("Every grid level must be greater than one")
    if tuple(sorted(set(grid_levels))) != tuple(grid_levels):
        raise ValueError("grid_levels must be strictly increasing and unique")

    target = min(int(maximum_points), int(points.shape[0]))
    minimum = points.min(axis=0).astype(np.float64)
    span = points.max(axis=0).astype(np.float64) - minimum
    span[span == 0.0] = 1.0
    normalized = ((points.astype(np.float64) - minimum) / span).astype(np.float32)
    normalized = np.clip(
        normalized,
        np.float32(0.0),
        np.nextafter(np.float32(1.0), np.float32(0.0)),
    )

    selected = np.zeros(points.shape[0], dtype=bool)
    ordered_chunks: list[np.ndarray] = []
    selected_count = 0
    for level in grid_levels:
        cells = np.floor(normalized * np.float32(level)).astype(np.int64)
        codes = cells[:, 0] + level * (cells[:, 1] + level * cells[:, 2])
        unique_codes, first_indices = np.unique(codes, return_index=True)
        new_mask = ~selected[first_indices]
        candidates = first_indices[new_mask]
        candidate_codes = unique_codes[new_mask]
        if candidates.size == 0:
            continue
        order = np.argsort(_splitmix64(candidate_codes), kind="stable")
        candidates = candidates[order]
        remaining = target - selected_count
        if candidates.size > remaining:
            candidates = candidates[:remaining]
        selected[candidates] = True
        ordered_chunks.append(candidates.astype(np.int64, copy=False))
        selected_count += int(candidates.size)
        if selected_count == target:
            break

    if selected_count < target:
        remaining_indices = np.flatnonzero(~selected)
        order = np.argsort(_splitmix64(remaining_indices), kind="stable")
        needed = target - selected_count
        ordered_chunks.append(remaining_indices[order[:needed]])

    result = np.concatenate(ordered_chunks) if ordered_chunks else np.empty(0, dtype=np.int64)
    if result.shape[0] != target or np.unique(result).shape[0] != target:
        raise RuntimeError("Progressive sampler did not produce an exact unique target")
    return result


def validate_quality_targets(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != len(QUALITY_LABELS):
        raise ValueError("Exactly three quality point targets are required")
    targets = tuple(int(value) for value in values)
    if targets[0] <= 0 or not (targets[0] < targets[1] < targets[2]):
        raise ValueError("Quality point targets must be positive and strictly increasing")
    return targets


def _rgb_strings(colors: np.ndarray) -> list[str]:
    return [f"rgb({red},{green},{blue})" for red, green, blue in colors]


def build_figure(
    points: np.ndarray,
    colors: np.ndarray,
    progressive_indices: np.ndarray,
    quality_targets: tuple[int, int, int],
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
        layer_colors = colors[layer_indices]
        traces.append(
            go.Scatter3d(
                x=layer_points[:, 0],
                y=layer_points[:, 1],
                z=layer_points[:, 2],
                mode="markers",
                marker={
                    "size": DEFAULT_POINT_SIZE,
                    "color": _rgb_strings(layer_colors),
                    "opacity": DEFAULT_OPACITY,
                },
                hoverinfo="skip",
                name=label,
                visible=layer_index < 2,
            )
        )

    figure = go.Figure(data=traces)
    axis = {
        "showbackground": True,
        "backgroundcolor": "#080b0f",
        "gridcolor": "#273849",
        "zerolinecolor": "#52677c",
        "tickfont": {"color": "#8fa2b5", "size": 11},
    }
    axis_title_font = {"color": "#aebdca", "size": 12}
    figure.update_layout(
        paper_bgcolor="#080b0f",
        plot_bgcolor="#080b0f",
        font={"color": "#d9e2ea", "family": "Segoe UI, sans-serif"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        uirevision="e26-pointcloud-inspector-v2",
        scene={
            "aspectmode": "data",
            "bgcolor": "#080b0f",
            "camera": {
                "eye": {"x": 1.55, "y": -1.55, "z": 1.05},
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
            },
            "xaxis": {**axis, "title": {"text": "x (m)", "font": axis_title_font}},
            "yaxis": {**axis, "title": {"text": "y (m)", "font": axis_title_font}},
            "zaxis": {**axis, "title": {"text": "z (m)", "font": axis_title_font}},
        },
    )
    return figure, actual_counts


def build_document(
    figure: go.Figure,
    *,
    title: str,
    source_point_count: int,
    actual_counts: dict[str, int],
) -> str:
    plot_html = to_html(
        figure,
        include_plotlyjs=True,
        full_html=False,
        div_id="pointcloud-viewer",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    metadata_json = json.dumps(
        {"source": int(source_point_count), "counts": actual_counts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; --panel: rgba(12, 17, 23, .94); --line: #2b3b49; --ink: #edf2f6; --muted: #92a5b7; --accent: #54c3ad; }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; background: #080b0f; color: var(--ink); font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; }}
    #viewer-shell, #pointcloud-viewer {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    .identity {{ position: fixed; top: 18px; left: 20px; z-index: 4; pointer-events: none; text-shadow: 0 1px 8px #000; }}
    .identity h1 {{ margin: 0 0 5px; font-size: 17px; font-weight: 650; letter-spacing: .01em; }}
    .identity p {{ margin: 0; color: #a9bac8; font-size: 12px; }}
    .panel {{ position: fixed; top: 16px; right: 16px; z-index: 5; width: 318px; max-height: calc(100vh - 32px); overflow: auto; padding: 16px; border: 1px solid var(--line); background: var(--panel); box-shadow: 0 18px 55px rgba(0,0,0,.35); backdrop-filter: blur(12px); }}
    .panel.collapsed {{ width: auto; padding: 8px; overflow: hidden; }}
    .panel.collapsed .panel-body, .panel.collapsed .panel-title {{ display: none; }}
    .panel-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .panel-title {{ font-size: 12px; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; }}
    button, input {{ font: inherit; }}
    button {{ min-height: 34px; border: 1px solid #344857; background: #121a22; color: #dce6ed; padding: 7px 10px; cursor: pointer; }}
    button:hover {{ border-color: #6f899d; background: #18242e; }}
    button.active {{ border-color: var(--accent); color: #dffff8; background: #15312d; }}
    .icon-button {{ width: 36px; padding: 6px; font-family: ui-monospace, Consolas, monospace; }}
    .group {{ border-top: 1px solid #23313c; margin-top: 14px; padding-top: 13px; }}
    .group-label {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; font-size: 12px; color: var(--muted); }}
    .value {{ color: var(--ink); font-family: ui-monospace, Consolas, monospace; }}
    .button-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }}
    .camera-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); cursor: ew-resize; }}
    .toggle-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
    .status {{ margin-top: 13px; padding: 10px; border: 1px solid #23313c; background: #0a0f14; color: #9fb0bd; font-size: 11px; line-height: 1.55; }}
    .status strong {{ color: #dfe9ef; font-weight: 600; }}
    .shortcut {{ margin: 10px 0 0; color: #708596; font-size: 10px; line-height: 1.5; }}
    .badge {{ color: var(--accent); }}
    @media (max-width: 720px) {{ .panel {{ top: auto; right: 10px; bottom: 10px; left: 10px; width: auto; max-height: 48vh; }} .identity {{ top: 10px; left: 12px; max-width: calc(100vw - 70px); }} }}
  </style>
</head>
<body>
  <main id="viewer-shell">
    {plot_html}
    <header class="identity">
      <h1>{title}</h1>
      <p>重力校准裁切点云 · 只读显示层 · 无补点 / 无补面 / 无 Mesh</p>
    </header>
    <aside class="panel" id="inspector" aria-label="点云显示控制">
      <div class="panel-head">
        <span class="panel-title">Point cloud inspector</span>
        <button class="icon-button" id="panel-toggle" title="收起控制面板" aria-label="收起控制面板">—</button>
      </div>
      <div class="panel-body">
        <section class="group">
          <div class="group-label"><span>显示密度</span><span class="value" id="quality-value">平衡</span></div>
          <div class="button-row" id="quality-buttons">
            <button data-quality="performance">流畅</button>
            <button class="active" data-quality="balanced">平衡</button>
            <button data-quality="detail">精细</button>
          </div>
        </section>
        <section class="group">
          <label class="group-label" for="point-size"><span>点大小</span><span class="value" id="point-size-value">1.00 px</span></label>
          <input id="point-size" type="range" min="0.25" max="3.00" step="0.05" value="1.00" />
        </section>
        <section class="group">
          <label class="group-label" for="point-opacity"><span>点透明度</span><span class="value" id="point-opacity-value">0.90</span></label>
          <input id="point-opacity" type="range" min="0.25" max="1.00" step="0.05" value="0.90" />
        </section>
        <section class="group">
          <div class="group-label"><span>标准视角</span><span class="value">Z ↑</span></div>
          <div class="camera-grid" id="camera-buttons">
            <button data-camera="front">正面</button><button data-camera="right">右侧</button><button data-camera="rear">背面</button>
            <button data-camera="left">左侧</button><button data-camera="top">顶视</button><button class="active" data-camera="oblique">斜视</button>
          </div>
        </section>
        <section class="group toggle-row">
          <button class="active" id="grid-toggle">网格：开</button>
          <button id="theme-toggle">背景：深</button>
        </section>
        <div class="status" id="status"></div>
        <p class="shortcut">滚轮缩放 · 左键旋转 · 右键平移 · 数字 1/2/3 切换密度 · [ / ] 调节点径 · G 切换网格</p>
      </div>
    </aside>
  </main>
  <script>
  (() => {{
    const meta = {metadata_json};
    const graph = document.getElementById("pointcloud-viewer");
    const qualityNames = {{ performance: "流畅", balanced: "平衡", detail: "精细" }};
    const visibility = {{ performance: [true, false, false], balanced: [true, true, false], detail: [true, true, true] }};
    const cameras = {{
      front: {{ eye: {{x: 0, y: -2.35, z: .35}}, up: {{x:0,y:0,z:1}}, center: {{x:0,y:0,z:0}} }},
      right: {{ eye: {{x: 2.35, y: 0, z: .35}}, up: {{x:0,y:0,z:1}}, center: {{x:0,y:0,z:0}} }},
      rear: {{ eye: {{x: 0, y: 2.35, z: .35}}, up: {{x:0,y:0,z:1}}, center: {{x:0,y:0,z:0}} }},
      left: {{ eye: {{x: -2.35, y: 0, z: .35}}, up: {{x:0,y:0,z:1}}, center: {{x:0,y:0,z:0}} }},
      top: {{ eye: {{x: 0, y: 0, z: 2.65}}, up: {{x:0,y:1,z:0}}, center: {{x:0,y:0,z:0}} }},
      oblique: {{ eye: {{x: 1.55, y: -1.55, z: 1.05}}, up: {{x:0,y:0,z:1}}, center: {{x:0,y:0,z:0}} }}
    }};
    let quality = "balanced";
    let grid = true;
    let light = false;
    const countText = n => new Intl.NumberFormat("zh-CN").format(n);
    const status = document.getElementById("status");
    function updateStatus() {{
      status.innerHTML = `<strong>${{qualityNames[quality]}}</strong> · 显示 ${{countText(meta.counts[quality])}} / ${{countText(meta.source)}} 点<br><span class="badge">空间分层抽样</span>仅改变预览密度，源 PLY 不变。`;
      document.getElementById("quality-value").textContent = qualityNames[quality];
    }}
    function setQuality(next) {{
      quality = next;
      Plotly.restyle(graph, {{visible: visibility[next]}}, [0, 1, 2]);
      document.querySelectorAll("[data-quality]").forEach(button => button.classList.toggle("active", button.dataset.quality === next));
      updateStatus();
    }}
    function setPointSize(value) {{
      const parsed = Number(value);
      Plotly.restyle(graph, {{"marker.size": parsed}}, [0, 1, 2]);
      document.getElementById("point-size").value = parsed.toFixed(2);
      document.getElementById("point-size-value").textContent = `${{parsed.toFixed(2)}} px`;
    }}
    function setOpacity(value) {{
      const parsed = Number(value);
      Plotly.restyle(graph, {{"marker.opacity": parsed}}, [0, 1, 2]);
      document.getElementById("point-opacity").value = parsed.toFixed(2);
      document.getElementById("point-opacity-value").textContent = parsed.toFixed(2);
    }}
    document.getElementById("quality-buttons").addEventListener("click", event => {{ if (event.target.dataset.quality) setQuality(event.target.dataset.quality); }});
    document.getElementById("point-size").addEventListener("input", event => setPointSize(event.target.value));
    document.getElementById("point-opacity").addEventListener("input", event => setOpacity(event.target.value));
    document.getElementById("camera-buttons").addEventListener("click", event => {{
      const name = event.target.dataset.camera;
      if (!name) return;
      Plotly.relayout(graph, {{"scene.camera": cameras[name]}});
      document.querySelectorAll("[data-camera]").forEach(button => button.classList.toggle("active", button.dataset.camera === name));
    }});
    document.getElementById("grid-toggle").addEventListener("click", event => {{
      grid = !grid;
      Plotly.relayout(graph, {{"scene.xaxis.showgrid": grid, "scene.yaxis.showgrid": grid, "scene.zaxis.showgrid": grid}});
      event.target.textContent = `网格：${{grid ? "开" : "关"}}`;
      event.target.classList.toggle("active", grid);
    }});
    document.getElementById("theme-toggle").addEventListener("click", event => {{
      light = !light;
      const background = light ? "#edf1f3" : "#080b0f";
      const gridColor = light ? "#c3ccd2" : "#273849";
      const fontColor = light ? "#26343e" : "#d9e2ea";
      Plotly.relayout(graph, {{paper_bgcolor: background, plot_bgcolor: background, font: {{color: fontColor}}, "scene.bgcolor": background, "scene.xaxis.backgroundcolor": background, "scene.yaxis.backgroundcolor": background, "scene.zaxis.backgroundcolor": background, "scene.xaxis.gridcolor": gridColor, "scene.yaxis.gridcolor": gridColor, "scene.zaxis.gridcolor": gridColor}});
      event.target.textContent = `背景：${{light ? "浅" : "深"}}`;
      event.target.classList.toggle("active", light);
    }});
    document.getElementById("panel-toggle").addEventListener("click", event => {{
      const panel = document.getElementById("inspector");
      panel.classList.toggle("collapsed");
      event.target.textContent = panel.classList.contains("collapsed") ? "+" : "—";
      event.target.title = panel.classList.contains("collapsed") ? "展开控制面板" : "收起控制面板";
    }});
    document.addEventListener("keydown", event => {{
      if (event.key === "1") setQuality("performance");
      if (event.key === "2") setQuality("balanced");
      if (event.key === "3") setQuality("detail");
      if (event.key.toLowerCase() === "g") document.getElementById("grid-toggle").click();
      if (event.key === "[") setPointSize(Math.max(.25, Number(document.getElementById("point-size").value) - .1));
      if (event.key === "]") setPointSize(Math.min(3, Number(document.getElementById("point-size").value) + .1));
    }});
    updateStatus();
  }})();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--quality-points",
        nargs=3,
        type=int,
        default=DEFAULT_QUALITY_TARGETS,
        metavar=("PERFORMANCE", "BALANCED", "DETAIL"),
    )
    args = parser.parse_args()
    quality_targets = validate_quality_targets(args.quality_points)
    if args.output_html.exists() or args.output_receipt.exists():
        raise FileExistsError("Output HTML and receipt must not already exist")

    input_hash_before = sha256(args.input_ply)
    points, colors = read_xyzrgb_ply(args.input_ply)
    maximum_points = min(points.shape[0], quality_targets[-1])
    sampled_indices = progressive_spatial_indices(points, maximum_points)
    figure, actual_counts = build_figure(points, colors, sampled_indices, quality_targets)
    document = build_document(
        figure,
        title=args.title,
        source_point_count=int(points.shape[0]),
        actual_counts=actual_counts,
    )
    if re.search(r"<script\b[^>]*\bsrc\s*=", document, flags=re.IGNORECASE):
        raise RuntimeError("Viewer contains an external script dependency")

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    with args.output_html.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    input_hash_after = sha256(args.input_ply)
    if input_hash_after != input_hash_before:
        raise RuntimeError("Source PLY changed while building the viewer")

    receipt = {
        "schema_version": "e26_pointcloud_interactive_viewer_receipt_v2",
        "status": "complete",
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__)),
        "input_ply": str(args.input_ply.resolve()),
        "input_ply_sha256": input_hash_before,
        "input_point_count": int(points.shape[0]),
        "display_point_counts": actual_counts,
        "maximum_embedded_point_count": int(maximum_points),
        "sampling": {
            "algorithm": "deterministic_progressive_multiscale_voxel_strata_v1",
            "grid_levels": list(SAMPLING_LEVELS),
            "nested_quality_prefixes": True,
            "source_order_even_index_sampling": False,
        },
        "viewer_defaults": {
            "quality": "balanced",
            "point_size_px": DEFAULT_POINT_SIZE,
            "opacity": DEFAULT_OPACITY,
            "background": "dark",
            "grid": True,
            "camera": "oblique_z_up",
        },
        "viewer_controls": [
            "quality_performance_balanced_detail",
            "point_size_0.25_to_3.00_px",
            "point_opacity_0.25_to_1.00",
            "six_camera_presets",
            "grid_toggle",
            "dark_light_background_toggle",
            "collapsible_inspector",
            "keyboard_shortcuts",
        ],
        "offline_self_contained": True,
        "external_script_count": 0,
        "output_html": str(args.output_html.resolve()),
        "output_html_bytes": args.output_html.stat().st_size,
        "output_html_sha256": sha256(args.output_html),
        "source_ply_unchanged": True,
        "source_ply_sha256_after": input_hash_after,
        "mesh_generation_count": 0,
        "synthetic_point_generation_count": 0,
        "crop_modification_count": 0,
        "render_boundary": "Display-only spatially stratified subsampling; source PLY and crop are unchanged.",
        "process_id": os.getpid(),
    }
    with args.output_receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

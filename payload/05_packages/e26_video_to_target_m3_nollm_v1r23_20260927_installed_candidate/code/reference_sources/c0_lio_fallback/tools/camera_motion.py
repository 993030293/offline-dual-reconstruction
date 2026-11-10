#!/usr/bin/env python
"""Camera motion proxy and preview extraction from the raw Bayer topic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_contract import COLOR_CONTRACT_VERSION, demosaic_rggb8_to_bgr
from mcap_audit import Cdr, McapIndex


def decode_bayer(payload: bytes) -> tuple[int, str, np.ndarray]:
    cdr = Cdr(payload)
    sec, nanosec, frame_id = cdr.header()
    height = cdr.u32()
    width = cdr.u32()
    encoding = cdr.string()
    cdr.u8()  # is_bigendian
    step = cdr.u32()
    data_length = cdr.u32()
    raw = np.frombuffer(
        payload,
        dtype=np.uint8,
        count=data_length,
        offset=cdr.offset,
    ).reshape(height, step)[:, :width]
    if encoding != "bayer_rggb8":
        raise ValueError(f"Unexpected encoding: {encoding}")
    return sec * 1_000_000_000 + nanosec, frame_id, raw


def range_stats(time_s: np.ndarray, values: np.ndarray, start: float, end: float) -> dict:
    selected = (time_s >= start) & (time_s <= end)
    segment = values[selected]
    return {
        "start_s": start,
        "end_s": end,
        "count": int(segment.size),
        "mean": float(np.mean(segment)),
        "median": float(np.median(segment)),
        "p95": float(np.percentile(segment, 95)),
        "max": float(np.max(segment)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    mcap = args.mcap.resolve()
    output_dir = args.output_dir.resolve()
    if mcap.parent == output_dir or mcap.parent in output_dir.parents:
        raise SystemExit("Refusing to write output inside the original bag directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    index = McapIndex(mcap)
    topic = "/sensing/camera/image_raw"
    count = index.message_count(topic)
    selected = list(range(0, count, args.stride))
    if selected[-1] != count - 1:
        selected.append(count - 1)

    times_ns: list[int] = []
    brightness: list[float] = []
    contrast: list[float] = []
    focus: list[float] = []
    frame_difference: list[float] = []
    previous: np.ndarray | None = None
    preview_targets_s = [2.0, 10.0, 20.0, 80.0, 159.5]
    previews_written: set[int] = set()

    first_time_ns = None
    for message_number, message in index.iter_selected_messages(topic, selected):
        stamp_ns, _, raw = decode_bayer(message["payload"])
        if first_time_ns is None:
            first_time_ns = stamp_ns
        relative_s = (stamp_ns - first_time_ns) / 1e9
        small = cv2.resize(
            raw,
            (raw.shape[1] // 8, raw.shape[0] // 8),
            interpolation=cv2.INTER_AREA,
        )
        times_ns.append(stamp_ns)
        brightness.append(float(np.mean(small)))
        contrast.append(float(np.std(small)))
        focus.append(float(cv2.Laplacian(small, cv2.CV_64F).var()))
        if previous is None:
            frame_difference.append(0.0)
        else:
            difference = cv2.absdiff(small, previous)
            frame_difference.append(float(np.mean(difference) / 255.0))
        previous = small

        for target_number, target_s in enumerate(preview_targets_s):
            if (
                target_number not in previews_written
                and abs(relative_s - target_s) <= args.stride / 50.0
            ):
                bgr = demosaic_rggb8_to_bgr(raw)
                preview_path = preview_dir / f"camera_{target_s:06.1f}s.png"
                encoded, buffer = cv2.imencode(".png", bgr)
                if not encoded:
                    raise RuntimeError(f"Failed to encode preview: {preview_path}")
                buffer.tofile(preview_path)
                previews_written.add(target_number)

    times = (np.asarray(times_ns, dtype=np.int64) - times_ns[0]) / 1e9
    differences = np.asarray(frame_difference, dtype=np.float64)
    brightness_array = np.asarray(brightness, dtype=np.float64)
    contrast_array = np.asarray(contrast, dtype=np.float64)
    focus_array = np.asarray(focus, dtype=np.float64)

    data = {
        "source": str(mcap),
        "color_contract": COLOR_CONTRACT_VERSION,
        "stride": args.stride,
        "sample_rate_hz": 50.0 / args.stride,
        "sample_count": int(times.size),
        "frame_difference": {
            "initial_0.5_11.5s": range_stats(times, differences, 0.5, 11.5),
            "motion_12_157s": range_stats(times, differences, 12.0, 157.0),
            "final_159_161s": range_stats(times, differences, 159.0, 161.0),
        },
        "brightness": {
            "mean": float(brightness_array.mean()),
            "min": float(brightness_array.min()),
            "max": float(brightness_array.max()),
        },
        "contrast": {
            "mean": float(contrast_array.mean()),
            "min": float(contrast_array.min()),
            "max": float(contrast_array.max()),
        },
        "focus_laplacian_variance": {
            "median": float(np.median(focus_array)),
            "p05": float(np.percentile(focus_array, 5)),
            "min": float(focus_array.min()),
        },
        "preview_files": sorted(path.name for path in preview_dir.glob("*.png")),
        "series": {
            "time_s": [float(value) for value in times],
            "frame_difference": [float(value) for value in differences],
            "brightness": [float(value) for value in brightness_array],
            "contrast": [float(value) for value in contrast_array],
            "focus": [float(value) for value in focus_array],
        },
    }

    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(times, differences)
    axes[0].set_ylabel("frame MAD / 255")
    axes[1].plot(times, brightness_array, label="brightness")
    axes[1].plot(times, contrast_array, label="contrast")
    axes[1].legend()
    axes[1].set_ylabel("pixel value")
    axes[2].plot(times, focus_array)
    axes[2].set_ylabel("Laplacian variance")
    axes[2].set_xlabel("camera header time from start [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "camera_quality.png", dpi=160)
    plt.close(figure)

    (output_dir / "camera_motion.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Camera 运动与图像质量抽样",
        "",
        f"- 抽样步长：每 {args.stride} 帧",
        f"- 等效频率：{data['sample_rate_hz']:.3f} Hz",
        f"- 抽样数：{data['sample_count']}",
        "",
        "## 相邻抽样帧差",
        "",
        "| 区间 | 均值 | 中位数 | P95 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in data["frame_difference"].items():
        report.append(
            f"| {name} | {item['mean']:.6f} | {item['median']:.6f} | "
            f"{item['p95']:.6f} | {item['max']:.6f} |"
        )
    report.extend(
        [
            "",
            f"- 亮度范围：{data['brightness']['min']:.2f}–{data['brightness']['max']:.2f}",
            f"- 对比度范围：{data['contrast']['min']:.2f}–{data['contrast']['max']:.2f}",
            f"- 清晰度 Laplacian 方差 P05：{data['focus_laplacian_variance']['p05']:.2f}",
            "",
            "相邻帧差只用于确认静止/运动区间，不等同于相机轨迹或精确时间偏移。",
            "",
        ]
    )
    (output_dir / "camera_motion.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(output_dir / "camera_motion.json")
    print(output_dir / "camera_motion.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare a color-agnostic visual SfM workspace from an indexed rig MCAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_TOPIC = "/sensing/camera/image_raw"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sfm_tools(tools_root: Path) -> tuple[Any, Any, Any, Any, str]:
    """Load MCAP decoders and the explicit frozen Bayer color contract."""

    resolved = tools_root.resolve(strict=True)
    sys.path.insert(0, str(resolved))
    from camera_motion import decode_bayer
    from color_contract import COLOR_CONTRACT_VERSION, demosaic_rggb8_to_bgr
    from mcap_audit import McapIndex, decode_camera_info

    return (
        McapIndex,
        decode_bayer,
        decode_camera_info,
        demosaic_rggb8_to_bgr,
        COLOR_CONTRACT_VERSION,
    )


def image_quality(image: np.ndarray) -> dict[str, float]:
    """Measure color-agnostic focus and exposure quality."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    brightness = float(np.mean(gray))
    score = float(np.log1p(max(0.0, sharpness)) - 2.0 * clipped)
    return {
        "sharpness": sharpness,
        "clipped_fraction": clipped,
        "mean_brightness": brightness,
        "quality_score": score,
    }


def select_temporally_complete(
    candidates: list[dict[str, Any]], target_count: int
) -> list[dict[str, Any]]:
    """Choose one best frame per temporal bin without using object color."""

    if target_count < 2:
        raise ValueError("target_count must be at least two")
    if len(candidates) < target_count:
        raise ValueError("candidate_count must not be smaller than target_count")
    boundaries = np.linspace(0, len(candidates), target_count + 1, dtype=np.int64)
    selected = []
    for index in range(target_count):
        group = candidates[int(boundaries[index]) : int(boundaries[index + 1])]
        if not group:
            raise RuntimeError(f"Empty temporal selection bin: {index}")
        selected.append(
            max(
                group,
                key=lambda item: (
                    float(item["quality_score"]),
                    float(item["sharpness"]),
                    -int(item["camera_frame"]),
                ),
            )
        )
    return selected


def run_colmap(executable: Path, arguments: list[str], log_path: Path) -> None:
    """Run one COLMAP command with the packaged CUDA runtime on PATH."""

    environment = os.environ.copy()
    bin_dir = executable.parent
    environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
    environment["QT_PLUGIN_PATH"] = str(bin_dir.parent / "plugins")
    command = [str(executable), *arguments]
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )


def write_contact_sheet(
    image_dir: Path,
    manifest: list[dict[str, Any]],
    output_path: Path,
    maximum_panels: int = 24,
) -> None:
    """Write a bounded visual QA sheet from selected images."""

    panel_count = min(maximum_panels, len(manifest))
    indices = np.linspace(0, len(manifest) - 1, panel_count, dtype=np.int64)
    panels = []
    for index in indices:
        item = manifest[int(index)]
        image = cv2.imread(str(image_dir / item["image_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_dir / item["image_name"])
        scale = 320.0 / image.shape[1]
        panel = cv2.resize(image, (320, max(1, round(image.shape[0] * scale))))
        text = (
            f"t={float(item['relative_s']):.1f}s "
            f"f={int(item['camera_frame'])} "
            f"S={float(item['sharpness']):.0f}"
        )
        cv2.rectangle(panel, (0, 0), (320, 25), (0, 0, 0), -1)
        cv2.putText(
            panel,
            text,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    height = max(panel.shape[0] for panel in panels)
    panels = [
        cv2.copyMakeBorder(
            panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT
        )
        for panel in panels
    ]
    columns = 4
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def extract_images(
    mcap_path: Path,
    tools_root: Path,
    output_dir: Path,
    candidate_count: int,
    image_count: int,
    start_fraction: float,
    end_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode and select temporally complete sharp frames."""

    (
        McapIndex,
        decode_bayer,
        decode_camera_info,
        demosaic_rggb8_to_bgr,
        color_contract_version,
    ) = load_sfm_tools(tools_root)
    index = McapIndex(mcap_path)
    total = int(index.message_count(CAMERA_TOPIC))
    start = int(round((total - 1) * start_fraction))
    end = int(round((total - 1) * end_fraction))
    if not 0 <= start < end < total:
        raise ValueError("Invalid start/end fraction")
    candidate_count = min(candidate_count, end - start + 1)
    candidate_frames = np.unique(
        np.linspace(start, end, candidate_count, dtype=np.int64)
    ).tolist()
    candidates = []
    first_stamp_ns: int | None = None
    for frame, message in index.iter_selected_messages(CAMERA_TOPIC, candidate_frames):
        stamp_ns, _, raw = decode_bayer(message["payload"])
        if first_stamp_ns is None:
            first_stamp_ns = int(stamp_ns)
        image = demosaic_rggb8_to_bgr(raw)
        candidates.append(
            {
                "camera_frame": int(frame),
                "camera_header_ns": int(stamp_ns),
                "relative_s": float((int(stamp_ns) - first_stamp_ns) / 1e9),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                **image_quality(image),
            }
        )
    selected = select_temporally_complete(candidates, image_count)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    selected_messages = dict(
        index.iter_selected_messages(
            CAMERA_TOPIC, [int(item["camera_frame"]) for item in selected]
        )
    )
    manifest = []
    origin_stamp = int(candidates[0]["camera_header_ns"])
    for item in selected:
        frame = int(item["camera_frame"])
        stamp_ns, _, raw = decode_bayer(selected_messages[frame]["payload"])
        image = demosaic_rggb8_to_bgr(raw)
        name = f"frame_{frame:05d}_{int(stamp_ns)}.jpg"
        encoded, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 97]
        )
        if not encoded:
            raise RuntimeError(f"Could not encode frame {frame}")
        image_path = image_dir / name
        buffer.tofile(image_path)
        manifest.append(
            {
                **item,
                "relative_s": float((int(stamp_ns) - origin_stamp) / 1e9),
                "image_name": name,
                "image_sha256": sha256_file(image_path),
            }
        )
    info_message = next(index.iter_messages("/sensing/camera/camera_info"))
    info = decode_camera_info(info_message["payload"])
    matrix = np.asarray(info["camera_matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(info["distortion"], dtype=np.float64)
    calibration_payload = {
        "camera_model": "OPENCV",
        "source_encoding": "bayer_rggb8",
        "decoded_array_order": "BGR",
        "color_contract": color_contract_version,
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
        "distortion": np.asarray(distortion, dtype=np.float64).reshape(-1).tolist(),
    }
    return manifest, calibration_payload


def largest_sparse_model(sparse_dir: Path) -> tuple[Path, Any]:
    """Return the COLMAP model with the most registered images."""

    import pycolmap

    models = []
    for directory in sparse_dir.iterdir():
        if directory.is_dir():
            reconstruction = pycolmap.Reconstruction(directory)
            models.append((reconstruction.num_reg_images(), directory, reconstruction))
    if not models:
        raise RuntimeError("COLMAP did not create a sparse model")
    _, path, reconstruction = max(models, key=lambda item: item[0])
    return path, reconstruction


def prepare_sparse(
    output_dir: Path,
    colmap: Path,
    calibration: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a fixed-intrinsics sparse model and undistorted dense workspace."""

    image_dir = output_dir / "images"
    database = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    dense_dir = output_dir / "dense_source_workspace"
    log_path = output_dir / "colmap.log"
    sparse_dir.mkdir(exist_ok=False)
    matrix = np.asarray(calibration["matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)
    camera_parameters = ",".join(
        str(float(value))
        for value in (
            matrix[0, 0],
            matrix[1, 1],
            matrix[0, 2],
            matrix[1, 2],
            distortion[0],
            distortion[1],
            distortion[2],
            distortion[3],
        )
    )
    run_colmap(
        colmap,
        [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "OPENCV",
            "--ImageReader.camera_params",
            camera_parameters,
            "--FeatureExtraction.use_gpu",
            "1",
            "--SiftExtraction.max_num_features",
            "12000",
            "--SiftExtraction.peak_threshold",
            "0.004",
        ],
        log_path,
    )
    run_colmap(
        colmap,
        [
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            "1",
        ],
        log_path,
    )
    run_colmap(
        colmap,
        [
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.multiple_models",
            "1",
            "--Mapper.min_model_size",
            "8",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.random_seed",
            "20260818",
        ],
        log_path,
    )
    model_dir, reconstruction = largest_sparse_model(sparse_dir)
    registered = {
        image.name for image in reconstruction.images.values() if image.has_pose
    }
    registered_manifest = [
        item for item in manifest if str(item["image_name"]) in registered
    ]
    registered_fraction = len(registered_manifest) / len(manifest)
    if len(registered_manifest) < 24 or registered_fraction < 0.5:
        raise RuntimeError(
            "Sparse registration is insufficient: "
            f"{len(registered_manifest)}/{len(manifest)}"
        )
    registered_path = output_dir / "registered_image_manifest.json"
    registered_path.write_text(
        json.dumps(registered_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_colmap(
        colmap,
        [
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            "1624",
            "--num_patch_match_src_images",
            "16",
        ],
        log_path,
    )
    return {
        "selected_image_count": len(manifest),
        "registered_image_count": len(registered_manifest),
        "registered_fraction": registered_fraction,
        "sparse_point_count": int(reconstruction.num_points3D()),
        "model_directory": str(model_dir.resolve()),
        "registered_manifest": str(registered_path.resolve()),
        "registered_manifest_sha256": sha256_file(registered_path),
        "dense_source_workspace": str(dense_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--sfm-tools", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=480)
    parser.add_argument("--image-count", type=int, default=180)
    parser.add_argument("--start-fraction", type=float, default=0.0)
    parser.add_argument("--end-fraction", type=float, default=1.0)
    parser.add_argument("--colmap", type=Path)
    parser.add_argument("--expected-mcap-sha256")
    parser.add_argument("--skip-full-mcap-hash", action="store_true")
    args = parser.parse_args()

    mcap = args.mcap.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if mcap.parent == output or mcap.parent in output.parents:
        raise ValueError("Output must not be inside the raw input directory")
    output.mkdir(parents=True)
    if args.expected_mcap_sha256 and not args.skip_full_mcap_hash:
        actual = sha256_file(mcap)
        if actual != args.expected_mcap_sha256.lower():
            raise ValueError("MCAP hash does not match the registered input")
    manifest, calibration = extract_images(
        mcap,
        args.sfm_tools,
        output,
        args.candidate_count,
        args.image_count,
        args.start_fraction,
        args.end_fraction,
    )
    manifest_path = output / "image_manifest.json"
    calibration_path = output / "camera_calibration.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_contact_sheet(output / "images", manifest, output / "contact_sheet.png")
    receipt: dict[str, Any] = {
        "schema_version": "e26_orbit_visual_mvs_workspace_receipt_v1",
        "status": "extracted",
        "mcap": {
            "path": str(mcap),
            "bytes": mcap.stat().st_size,
            "registered_sha256": args.expected_mcap_sha256,
            "full_hash_recomputed": not args.skip_full_mcap_hash,
        },
        "selection": {
            "candidate_count": args.candidate_count,
            "selected_image_count": len(manifest),
            "start_fraction": args.start_fraction,
            "end_fraction": args.end_fraction,
            "color_fields_used": [],
        },
        "color_contract": calibration["color_contract"],
        "image_manifest_sha256": sha256_file(manifest_path),
        "camera_calibration_sha256": sha256_file(calibration_path),
        "contact_sheet_sha256": sha256_file(output / "contact_sheet.png"),
    }
    if args.colmap is not None:
        receipt["sparse"] = prepare_sparse(
            output, args.colmap.resolve(strict=True), calibration, manifest
        )
        receipt["status"] = "sparse_and_undistorted_workspace_complete"
    receipt["scientific_boundary"] = (
        "Frame selection is color-agnostic and temporally complete. Sparse and dense "
        "visual geometry use an arbitrary similarity frame until separately aligned "
        "to metric LiDAR; this stage does not apply an object mask."
    )
    receipt_path = output / "workspace_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

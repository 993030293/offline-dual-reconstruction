"""Inspect public ground truth in an evaluation-only namespace."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REVISION = "03f4382308333aa70c3253f12acd3fbf0c7c4a15"
SEQUENCE = "2024-03-13-observatory-quarter-01"
DATA = ROOT / "03_data/raw/public_benchmarks/oxford_spires" / REVISION
BAG = Path("E:/lidar_reconstruction_video_data/raw_recordings/public_benchmarks/oxford_spires") / REVISION / "raw/observatory/1710338090_2024-03-13-13-54-51_0.bag"
TLS = DATA / "evaluation/observatory/merged-clouds-1cm-no-colour.pcd"


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inspect_tls(path):
    fields = {}
    header = []
    with path.open("rb") as stream:
        for _ in range(40):
            raw = stream.readline(4096)
            if not raw or len(raw) >= 4096:
                raise ValueError("Unexpected PCD header")
            line = raw.decode("ascii").strip()
            header.append(line)
            if line and not line.startswith("#"):
                key, *values = line.split()
                fields[key] = values
            if line.startswith("DATA "):
                offset = stream.tell()
                break
        else:
            raise ValueError("PCD data marker absent")
    if fields["DATA"] != ["binary"] or fields["FIELDS"] != ["x", "y", "z", "_"]:
        raise ValueError("Unsupported TLS PCD layout")
    if fields["SIZE"] != ["4", "4", "4", "1"] or fields["COUNT"] != ["1", "1", "1", "4"] or fields["TYPE"] != ["F", "F", "F", "U"]:
        raise ValueError("Unexpected TLS point stride")
    count = int(fields["POINTS"][0])
    stride = 16
    # Binary points start immediately after DATA; file allocation can leave zero bytes at the end.
    payload_start = offset
    payload_end = offset + count * stride
    if path.stat().st_size < payload_end:
        raise ValueError("TLS payload is shorter than the declared point count")
    trailing_bytes = path.stat().st_size - payload_end
    if trailing_bytes > 4096:
        raise ValueError("Unexpectedly large trailing PCD allocation")
    with path.open("rb") as stream:
        stream.seek(payload_end)
        trailing = stream.read(trailing_bytes)
    if any(trailing):
        raise ValueError("Nonzero content after the declared binary point payload")
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [0, 4, 8], "itemsize": stride})
    data = np.memmap(path, dtype=dtype, mode="r", offset=payload_start, shape=(count,))
    indices = np.linspace(0, count - 1, min(count, 250_000), dtype=np.int64)
    sample = np.column_stack([data[name][indices] for name in ("x", "y", "z")])
    finite = np.isfinite(sample).all(axis=1)
    if finite.mean() < .99:
        raise ValueError("Too many nonfinite points in the bounded TLS sample")
    xyz = sample[finite]
    return {"path": str(path), "bytes": path.stat().st_size, "declared_points": count,
            "point_stride_bytes": stride, "header_data_offset": offset, "payload_offset": payload_start,
            "verified_zero_trailing_bytes": trailing_bytes,
            "sample_method": "deterministic_uniform_record_index_sample_not_full_cloud_validation",
            "sample_count": len(sample), "finite_sample_count": int(finite.sum()),
            "sample_bounds": [xyz.min(0).tolist(), xyz.max(0).tolist()],
            "coordinate_unit": "official_metric_map_contract_not_independent_unit_measurement",
            "role": "evaluation_only_gt_geometry", "header": header}, xyz


def fetch_tree(relative):
    url = f"https://huggingface.co/api/datasets/ori-drs/oxford_spires_dataset/tree/{REVISION}/{relative}?recursive=true&expand=false"
    try:
        with urlopen(Request(url, headers={"User-Agent": "Box2-GT-audit/2"}), timeout=25) as response:
            items = json.load(response)
            link = response.headers.get("Link")
        return {"status": "listed", "source": url, "items": items,
                "pagination_header": link, "complete_listing_confirmed": not bool(link)}
    except (HTTPError, URLError, TimeoutError) as error:
        return {"status": "failed_fetch", "source": url, "error": str(error)}


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    tls_receipt, sample = inspect_tls(TLS)
    np.savez_compressed(output / "tls_uniform_sample_evaluation_only.npz", xyz=sample)
    raw_receipt = json.loads((ROOT / "04_outputs/public_all_cam0_parallel_preflight_20260930/raw_bag_receipt_e.json").read_text(encoding="utf-8"))
    if BAG.stat().st_size != raw_receipt["bytes"]:
        raise ValueError("The current public raw bag differs in size from the verified cache")
    calibration_files = [DATA / "calibration" / name for name in ("README.md", "cam-lidar-imu.yaml", "cam0.yaml", "cam1.yaml", "cam2.yaml", "imu.yaml")]
    if not all(path.is_file() for path in calibration_files):
        raise ValueError("Public sensor calibration is incomplete")
    result = {"status": "local_geometry_and_raw_inputs_available_trajectory_check_pending",
        "dataset": "Oxford Spires", "revision": REVISION, "sequence": SEQUENCE,
        "verified_current_raw_bag": {"path": str(BAG), "bytes": BAG.stat().st_size,
            "full_sha256_receipt": str(ROOT / "04_outputs/public_all_cam0_parallel_preflight_20260930/raw_bag_receipt_e.json"),
            "full_hash_recomputed_this_audit": False, "source_role": "raw_measurement"},
        "tls": tls_receipt, "calibration_files": [str(path) for path in calibration_files],
        "official_GT_definition_source": "https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/",
        "official_trajectory_convention": "T_world_base_frame; convert to camera using sensor extrinsic and timestamp association",
        "existing_raw_LIO_path": str(ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/00_raw_lio/raw_lio_receipt.json"),
        "existing_raw_LIO_role": "estimated_FAST_LIO_not_GT",
        "Box2_GT": False, "SfM_or_Gaussian_accuracy_evaluated": False,
        "GT_allowed_for_initial_frame_selection": False, "GT_allowed_for_initialization_or_matching": False,
        "use": "External Oxford-only pose and geometry evaluation, with locked coordinate, time, alignment, overlap and scale rules",
        "limits": ["Oxford and Box2 are different recordings", "TLS map is not a colour or image reference",
                   "Do not use official COLMAP or VILENS estimates as independent trajectory GT",
                   "Sample bounds validate readability, not reconstruction accuracy or full map coverage"],
        "checked_unix": time.time()}
    dump(output / "LOCAL_GT_AUDIT.json", result)
    print(json.dumps({"local_gt": {"points": tls_receipt["declared_points"], "sample": tls_receipt["sample_count"],
          "bounds": tls_receipt["sample_bounds"]}, "raw_bag_exists": BAG.is_file()}), flush=True)
    listing = fetch_tree(f"sequences/{SEQUENCE}/processed")
    dump(output / "OFFICIAL_PROCESSED_TREE.json", listing)
    print(json.dumps(listing, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output.resolve())

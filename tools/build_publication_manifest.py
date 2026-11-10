"""Build source-copy and publication manifests for the GitHub candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


NESTED_PACKAGE = "e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
SOURCE_ROOTS = (
    Path("payload/02_code"),
    Path("payload/05_packages") / NESTED_PACKAGE,
)
EXCLUDED_NAMES = {
    "OFFLINE_RUNTIME.local.json",
    "local_runtime.json",
    "INSTALL_RECEIPT.json",
    "INSTALL_VERIFY.json",
}
GENERATED_OUTPUTS = {
    "PUBLICATION_MANIFEST.json",
    "PUBLICATION_VERIFY_RESULT.json",
}
BASELINE_DELTA = Path("provenance/BASELINE_DELTA.json")
ALLOWED_NATIVE_RUNTIME_FILES = {
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/RUNTIME_MANIFEST.json",
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001/b0/native_bayer_gray.dll",
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/qc001/b0/native_planar_corner.dll",
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/vector_gray_stats_001/b0/native_vector_gray_stats.dll",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_path(path: Path) -> Path:
    """Use the Windows extended-length form without changing relative manifests."""
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\" + resolved)
    return Path(resolved)


def is_excluded_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    lowered_parts = {part.lower() for part in relative.parts}
    return (
        "__pycache__" in lowered_parts
        or "evidence" in lowered_parts
        or path.suffix.lower() == ".pyc"
        or path.name in EXCLUDED_NAMES
    )


def is_local_only_path(relative: str, is_directory: bool = False) -> bool:
    parts = Path(relative).parts
    if "__pycache__" in parts or relative.lower().endswith(".pyc"):
        return True
    if relative == ".git" or relative.startswith(".git/"):
        return True
    if relative == "payload/05_envs" or relative.startswith("payload/05_envs/"):
        return True
    if relative == "config/runtime.local.json" or relative.endswith("/config/local_runtime.json"):
        return True
    output_root = "payload/04_outputs"
    if relative == output_root:
        return False
    if relative.startswith(output_root + "/"):
        if relative in ALLOWED_NATIVE_RUNTIME_FILES:
            return False
        if is_directory and any(path.startswith(relative + "/") for path in ALLOWED_NATIVE_RUNTIME_FILES):
            return False
        return True
    return False


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for folder, directories, names in os.walk(root, followlinks=False):
        folder_path = Path(folder)
        retained_directories = []
        for name in sorted(directories):
            path = folder_path / name
            relative = path.relative_to(root).as_posix()
            if not is_local_only_path(relative, is_directory=True):
                retained_directories.append(name)
        directories[:] = retained_directories
        for name in sorted(names):
            path = folder_path / name
            relative = path.relative_to(root).as_posix()
            if relative in GENERATED_OUTPUTS or is_local_only_path(relative):
                continue
            files.append(path)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def load_baseline_delta(candidate_root: Path) -> dict:
    path = candidate_root / BASELINE_DELTA
    if not path.is_file():
        raise FileNotFoundError(f"Missing declared baseline delta: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_source_copy_map(candidate_root: Path, source_package: Path) -> dict:
    copied: list[dict] = []
    missing: list[str] = []
    unexpected_hashes: list[dict] = []

    for source_root_relative in SOURCE_ROOTS:
        original_root = source_package / source_root_relative
        if not original_root.is_dir():
            raise FileNotFoundError(f"Missing accepted source root: {original_root}")
        for original in sorted(original_root.rglob("*")):
            if not original.is_file() or is_excluded_source(original, original_root):
                continue
            relative = original.relative_to(source_package)
            candidate = candidate_root / relative
            relative_text = relative.as_posix()
            original_hash = sha256_file(original)
            if not candidate.is_file():
                missing.append(relative_text)
                continue
            candidate_hash = sha256_file(candidate)
            if candidate_hash != original_hash:
                unexpected_hashes.append(
                    {
                        "path": relative_text,
                        "accepted_sha256": original_hash,
                        "candidate_sha256": candidate_hash,
                    }
                )
            copied.append(
                {
                    "path": relative_text,
                    "bytes": candidate.stat().st_size,
                    "sha256": candidate_hash,
                }
            )

    copied_paths = {entry["path"] for entry in copied}
    extra_source_files: list[dict] = []
    for source_root_relative in SOURCE_ROOTS:
        candidate_source_root = candidate_root / source_root_relative
        for candidate in sorted(candidate_source_root.rglob("*")):
            if not candidate.is_file() or is_excluded_source(candidate, candidate_source_root):
                continue
            relative_text = candidate.relative_to(candidate_root).as_posix()
            if relative_text not in copied_paths:
                extra_source_files.append(
                    {
                        "path": relative_text,
                        "bytes": candidate.stat().st_size,
                        "sha256": sha256_file(candidate),
                    }
                )

    delta = load_baseline_delta(candidate_root)
    declared_modified = {entry["path"]: entry for entry in delta.get("modified_files", [])}
    declared_added = {entry["path"]: entry for entry in delta.get("added_files", [])}
    observed_modified = {entry["path"]: entry for entry in unexpected_hashes}
    observed_added = {entry["path"]: entry for entry in extra_source_files}
    delta_errors: list[dict] = []
    if set(observed_modified) != set(declared_modified):
        delta_errors.append(
            {
                "kind": "modified_path_set_mismatch",
                "declared": sorted(declared_modified),
                "observed": sorted(observed_modified),
            }
        )
    if set(observed_added) != set(declared_added):
        delta_errors.append(
            {
                "kind": "added_path_set_mismatch",
                "declared": sorted(declared_added),
                "observed": sorted(observed_added),
            }
        )
    for path, observed in observed_modified.items():
        declared = declared_modified.get(path)
        if declared is None:
            continue
        if (
            declared.get("base_sha256") != observed["accepted_sha256"]
            or declared.get("candidate_sha256") != observed["candidate_sha256"]
        ):
            delta_errors.append({"kind": "modified_hash_mismatch", "path": path})
    for path, observed in observed_added.items():
        declared = declared_added.get(path)
        if declared is None:
            continue
        if declared.get("candidate_sha256") != observed["sha256"]:
            delta_errors.append({"kind": "added_hash_mismatch", "path": path})

    status = (
        "accepted_base_plus_declared_delta"
        if not missing and not delta_errors
        else "source_copy_mismatch"
    )
    source_files = copied + list(observed_added.values())
    source_files.sort(key=lambda entry: entry["path"])
    return {
        "schema": "offline_dual_source_copy_map_v2",
        "status": status,
        "accepted_package_id": "offline_dual_reconstruction_v1_20261014",
        "historical_accepted_source_version": "bcf197d198d6e9ff2aef2d2dd7abfb6a025cb3f06df4739beb1a4d4190dc3dfd",
        "candidate_relation": "accepted_base_plus_declared_delta",
        "declared_delta": BASELINE_DELTA.as_posix(),
        "exclusions": [
            "Python bytecode and __pycache__",
            "directories named evidence",
            "OFFLINE_RUNTIME.local.json",
            "config/local_runtime.json",
            "INSTALL_RECEIPT.json and INSTALL_VERIFY.json",
        ],
        "files": source_files,
        "missing": missing,
        "declared_modified_files": unexpected_hashes,
        "declared_added_files": extra_source_files,
        "delta_errors": delta_errors,
    }


def source_tree_version(source_map: dict) -> str:
    rows = [
        {"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
        for entry in source_map["files"]
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    args = parser.parse_args()

    candidate_root = native_path(args.root)
    source_package = native_path(args.source_package)
    source_map = build_source_copy_map(candidate_root, source_package)
    source_map_path = candidate_root / "provenance/SOURCE_COPY_MAP.json"
    source_map_path.parent.mkdir(parents=True, exist_ok=True)
    source_map_path.write_text(
        json.dumps(source_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if source_map["status"] != "accepted_base_plus_declared_delta":
        raise RuntimeError("Candidate source tree does not match the declared accepted-base delta")

    entries = []
    total_bytes = 0
    for path in collect_files(candidate_root):
        relative = path.relative_to(candidate_root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        entries.append({"path": relative, "bytes": size, "sha256": sha256_file(path)})

    manifest = {
        "schema": "offline_dual_publication_manifest_v2",
        "candidate_id": "offline_dual_reconstruction_github_v0_1_20261110",
        "release_stage": json.loads(
            (candidate_root / BASELINE_DELTA).read_text(encoding="utf-8")
        )["candidate_stage"],
        "historical_accepted_source_version": "bcf197d198d6e9ff2aef2d2dd7abfb6a025cb3f06df4739beb1a4d4190dc3dfd",
        "candidate_algorithm_source_version": source_tree_version(source_map),
        "candidate_algorithm_file_count": len(source_map["files"]),
        "manifest_self_excluded": True,
        "generated_verify_result_excluded": True,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    (candidate_root / "PUBLICATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "built", "file_count": len(entries), "total_bytes": total_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

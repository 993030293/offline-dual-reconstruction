"""Verify the source-complete GitHub candidate without network access."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path


GENERATED_OUTPUTS = {
    "PUBLICATION_MANIFEST.json",
    "PUBLICATION_VERIFY_RESULT.json",
}
LOCAL_ONLY_NAMES = {
    "OFFLINE_RUNTIME.local.json",
    "local_runtime.json",
    "INSTALL_RECEIPT.json",
    "INSTALL_VERIFY.json",
    "PACKAGE_VERIFY_RESULT.json",
    "PUBLICATION_PUBLIC_GATE_RESULT.json",
    "RUNTIME_BINDING_RECEIPT.json",
}
BANNED_SUFFIXES = {
    ".bag",
    ".ckpt",
    ".db3",
    ".glb",
    ".gltf",
    ".las",
    ".laz",
    ".mcap",
    ".obj",
    ".onnx",
    ".pcd",
    ".ply",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
}
BANNED_COMPONENTS = {
    "04_outputs",
    "05_envs",
    "__pycache__",
    "evidence",
    "receipts",
}
ALLOWED_NATIVE_RUNTIME_FILES = {
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/RUNTIME_MANIFEST.json",
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001/b0/native_bayer_gray.dll",
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/qc001/b0/native_planar_corner.dll",
    "payload/04_outputs/gaussian_pipeline_optimization_v3_20261005/vector_gray_stats_001/b0/native_vector_gray_stats.dll",
}
REQUIRED_FILES = {
    "BIND_RUNTIME.ps1",
    "CHANGELOG.md",
    "PREFLIGHT.ps1",
    "README.md",
    "RUN.ps1",
    "SMOKE_SENSOR.ps1",
    "THIRD_PARTY.md",
    "VERIFY_PACKAGE.ps1",
    "config/runtime.binding.json",
    "docs/ALGORITHM_COMPLETENESS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/VALIDATION_STATUS.md",
    "licenses/GSPLAT_APACHE_2.0.txt",
    "licenses/INRIA_3DGS_LICENSE.md",
    "licenses/PGSR_LICENSE.md",
    "patches/pgsr/0001-projective-validity-and-alpha-output.patch",
    "provenance/ACCEPTED_SOURCE_IDENTITY.json",
    "provenance/BASELINE_DELTA.json",
    "provenance/EXTERNAL_SOURCE_LOCK.json",
    "provenance/PGSR_PATCH_VERIFY.json",
    "provenance/SOURCE_COPY_MAP.json",
}
SECRET_PATTERNS = {
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
PRIVACY_PATTERNS = {
    "windows_user_path": re.compile(r"C:[/\\]Users[/\\][^/\\\s\"']+", re.IGNORECASE),
    "codex_session_path": re.compile(r"\.codex[/\\]sessions", re.IGNORECASE),
}
KNOWN_NONPRODUCTION_SYNTAX_ARTIFACTS = {
    "payload/02_code/gaussian_pipeline_optimization_v3_20261005/review_array_match_negative.py",
    "payload/02_code/gaussian_pipeline_optimization_v3_20261005/review_native_sift_process_acceptance.py",
    "payload/02_code/gaussian_pipeline_optimization_v3_20261005/run_sift_process_campaign_002.py",
    "payload/02_code/gaussian_pipeline_optimization_v3_20261005/run_streamed_gaussian_ply_campaign.py",
}
TEXT_SUFFIXES = {
    ".c",
    ".cjs",
    ".cpp",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".patch",
    ".ps1",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_path(path: Path) -> Path:
    """Use the Windows extended-length form for deep source trees."""
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\" + resolved)
    return Path(resolved)


def is_reparse_point(path: Path) -> bool:
    info = os.lstat(path)
    attributes = getattr(info, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & flag)


def is_local_only_path(relative: str, is_directory: bool = False) -> bool:
    """Exclude ignored machine bindings while retaining the four frozen runtime files."""
    parts = Path(relative).parts
    if Path(relative).name in LOCAL_ONLY_NAMES:
        return True
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


def candidate_paths(root: Path):
    for folder, directories, files in os.walk(root, followlinks=False):
        folder_path = Path(folder)
        retained_directories = []
        for name in sorted(directories):
            path = folder_path / name
            relative = path.relative_to(root).as_posix()
            if is_local_only_path(relative, is_directory=True):
                continue
            retained_directories.append(name)
            yield path
        directories[:] = retained_directories
        for name in sorted(files):
            path = folder_path / name
            relative = path.relative_to(root).as_posix()
            if not is_local_only_path(relative):
                yield path


def actual_files(root: Path) -> set[str]:
    result = set()
    for path in candidate_paths(root):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in GENERATED_OUTPUTS:
            result.add(relative)
    return result


def scan_text(path: Path) -> tuple[list[str], list[str]]:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5 * 1024 * 1024:
        return [], []
    text = path.read_text(encoding="utf-8", errors="ignore")
    secrets = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    privacy = [name for name, pattern in PRIVACY_PATTERNS.items() if pattern.search(text)]
    return secrets, privacy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public", action="store_true", help="Fail on machine-specific user paths")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = native_path(args.root)
    manifest_path = root / "PUBLICATION_MANIFEST.json"
    source_map_path = root / "provenance/SOURCE_COPY_MAP.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))

    errors: list[dict] = []
    warnings: list[dict] = []
    expected = {entry["path"]: entry for entry in manifest["files"]}
    observed = actual_files(root)

    missing = sorted(set(expected) - observed)
    extra = sorted(observed - set(expected))
    if missing:
        errors.append({"kind": "manifest_missing", "paths": missing})
    if extra:
        errors.append({"kind": "manifest_extra", "paths": extra})

    hash_mismatches = []
    size_mismatches = []
    for relative, entry in expected.items():
        path = root / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != entry["bytes"]:
            size_mismatches.append(relative)
        if digest != entry["sha256"]:
            hash_mismatches.append(relative)
    if size_mismatches:
        errors.append({"kind": "size_mismatch", "paths": size_mismatches})
    if hash_mismatches:
        errors.append({"kind": "hash_mismatch", "paths": hash_mismatches})

    required_missing = sorted(path for path in REQUIRED_FILES if not (root / path).is_file())
    if required_missing:
        errors.append({"kind": "required_file_missing", "paths": required_missing})

    banned_files = []
    reparse_points = []
    secret_hits = []
    privacy_hits = []
    python_files = 0
    python_syntax_errors = []
    for path in candidate_paths(root):
        relative = path.relative_to(root).as_posix()
        if is_reparse_point(path):
            reparse_points.append(relative)
        if not path.is_file():
            continue
        lowered_parts = {part.lower() for part in path.relative_to(root).parts}
        if (
            relative not in ALLOWED_NATIVE_RUNTIME_FILES
            and (path.suffix.lower() in BANNED_SUFFIXES or lowered_parts & BANNED_COMPONENTS)
        ):
            banned_files.append(relative)
        secrets, privacy = scan_text(path)
        secret_hits.extend({"path": relative, "pattern": value} for value in secrets)
        privacy_hits.extend({"path": relative, "pattern": value} for value in privacy)
        if path.suffix.lower() == ".py":
            python_files += 1
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except (SyntaxError, UnicodeError) as error:
                python_syntax_errors.append({"path": relative, "error": str(error)})

    if banned_files:
        errors.append({"kind": "banned_payload", "paths": sorted(set(banned_files))})
    if reparse_points:
        errors.append({"kind": "reparse_point", "paths": sorted(set(reparse_points))})
    if secret_hits:
        errors.append({"kind": "possible_secret", "hits": secret_hits})
    if python_files == 0:
        errors.append({"kind": "python_syntax_check_empty"})
    unexpected_syntax_errors = [
        item for item in python_syntax_errors if item["path"] not in KNOWN_NONPRODUCTION_SYNTAX_ARTIFACTS
    ]
    known_syntax_artifacts = [
        item for item in python_syntax_errors if item["path"] in KNOWN_NONPRODUCTION_SYNTAX_ARTIFACTS
    ]
    if unexpected_syntax_errors:
        errors.append({"kind": "python_syntax_error", "hits": unexpected_syntax_errors})
    if known_syntax_artifacts:
        warnings.append({"kind": "known_nonproduction_python_syntax_artifact", "hits": known_syntax_artifacts})
    if privacy_hits:
        target = errors if args.public else warnings
        target.append({"kind": "machine_specific_path", "hits": privacy_hits})

    if source_map.get("status") != "accepted_base_plus_declared_delta":
        errors.append({"kind": "source_copy_status", "value": source_map.get("status")})
    source_hash_mismatches = []
    for entry in source_map.get("files", []):
        path = root / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            source_hash_mismatches.append(entry["path"])
    if source_hash_mismatches:
        errors.append({"kind": "accepted_source_mismatch", "paths": source_hash_mismatches})

    if not (root / "LICENSE").exists() and not (root / "LICENSE.md").exists():
        warnings.append({"kind": "project_license_not_selected"})

    result = {
        "schema": "offline_dual_publication_verify_result_v1",
        "status": "passed" if not errors else "failed",
        "mode": "public" if args.public else "private",
        "file_count": len(observed),
        "python_file_count": python_files,
        "python_syntax_valid_count": python_files - len(python_syntax_errors),
        "known_nonproduction_python_syntax_artifact_count": len(known_syntax_artifacts),
        "errors": errors,
        "warnings": warnings,
    }
    output = native_path(args.output) if args.output else root / "PUBLICATION_VERIFY_RESULT.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

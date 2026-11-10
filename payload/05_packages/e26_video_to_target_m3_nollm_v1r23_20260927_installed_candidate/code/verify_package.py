"""Verify package structure, JSON readability, Python syntax, and no-LLM source boundaries."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


REQUIRED = (
    "README.md",
    "ALGORITHM_SCOPE.md",
    "FAILURE_REPAIR_MATRIX.md",
    "REPAIR_RESULT.md",
    "INSTALL.ps1",
    "VERIFY_INSTALLATION.ps1",
    "RUN_UNATTENDED.ps1",
    "RUN_UPSTREAM_S00_S08.ps1",
    "RUN_FROM_S07_M3.ps1",
    "RUN_RELIABLE_LIDAR_S00_S08.ps1",
    "RUN_LEGACY_PGSR_S00_S08.ps1",
    "RUN_S08_PREVIEW.ps1",
    "contracts/NO_LLM_CONTRACT.json",
    "config/runtime.example.json",
    "config/ENVIRONMENT_LOCK.json",
    "config/METRIC_ALIGNMENT_POLICY.json",
    "code/runtime/build_s08_preview_input.py",
    "code/finalize_integrated_run.py",
    "code/write_pipeline_failure_receipt.py",
    "tests/test_s08_runtime.py",
    "tests/test_roi_derivation.py",
    "ground/config.json",
    "ground/code/ground_target_extraction.py",
    "ground/code/m3_ground.py",
    "ground/code/extract_m3_target.py",
    "ground/code/audit_m3_delivery.py",
    "ground/code/viewer/build_pointcloud_interactive_v2.py",
    "ground/code/viewer/build_pointcloud_interactive_v3.py",
    "ground/code/viewer/validate_pointcloud_interactive_v3.py",
)


def validate_runtime_caches(root: Path) -> int:
    caches = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and (path.suffix == ".pyc" or "__pycache__" in path.parts)
    )
    if not caches:
        return 0
    invalid = [
        path.as_posix()
        for path in caches
        if path.suffix != ".pyc" or "__pycache__" not in path.parts
    ]
    install_path = root / "INSTALL_RECEIPT.json"
    if invalid or not install_path.is_file():
        raise RuntimeError(
            f"compiled Python cache files are not allowed: "
            f"{invalid or [str(path) for path in caches[:10]]}"
        )
    installed = json.loads(install_path.read_text(encoding="utf-8-sig"))
    if (
        installed.get("schema_version") != "e26_install_receipt_v1"
        or installed.get("status") != "installed_and_runtime_verified"
    ):
        raise RuntimeError("Python caches require a verified installed-package receipt")
    return len(caches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)

    missing = [relative for relative in REQUIRED if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"missing required package files: {missing}")
    runtime_cache_file_count = validate_runtime_caches(root)

    python_files = sorted(root.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    json_files = sorted(root.rglob("*.json"))
    for path in json_files:
        if path.name in {"MANIFEST.json", "local_runtime.json"}:
            continue
        json.loads(path.read_text(encoding="utf-8-sig"))

    forbidden = ("import " + "openai", "from " + "openai", "import " + "anthropic")
    violations = []
    for path in sorted((*root.rglob("*.py"), *root.rglob("*.ps1"))):
        text = path.read_text(encoding="utf-8-sig").lower()
        if any(token in text for token in forbidden):
            violations.append(path.relative_to(root).as_posix())
    if violations:
        raise RuntimeError(f"runtime LLM client imports found: {violations}")

    contract = json.loads((root / "contracts" / "NO_LLM_CONTRACT.json").read_text(encoding="utf-8"))
    for key in (
        "runtime_language_model_used",
        "runtime_visual_language_model_used",
        "runtime_human_visual_decision_used",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"no-LLM contract does not set {key}=false")
    if contract.get("terminal_stage") != "M3_ground_removed_target_offline_preview":
        raise RuntimeError("no-LLM contract does not name the M3 terminal stage")
    print(
        json.dumps(
            {
                "status": "pass",
                "python_file_count": len(python_files),
                "json_file_count": len(json_files),
                "runtime_llm_client_imports": 0,
                "runtime_cache_file_count": runtime_cache_file_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

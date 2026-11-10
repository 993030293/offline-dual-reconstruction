"""Check native-text delivery, evidence links and first-pass metric identities."""

import argparse
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

from common import configure_stdout, read_json, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    code = Path(__file__).resolve().parent
    documents = [run / "RUN_REPORT.md", run / "02_same_180_ba/BA_COMPARISON.md", code / "README.md"]
    linked = []
    content = {}
    for document in documents:
        body = document.read_text(encoding="utf-8")
        require("\ufffd" not in body and body.startswith("# "), f"Invalid native text: {document}")
        content[document.name] = body
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
            if target.startswith(("https://", "http://")):
                continue
            resolved = (document.parent / target.split("#", 1)[0]).resolve()
            require(resolved.exists(), f"Broken local link: {document}: {label} -> {target}")
            linked.append(str(resolved))
    final = read_json(run / "FINAL_RECEIPT.json")
    report = content["RUN_REPORT.md"]
    require(final["original_inputs_preserved"], "Frozen source identity verification failed")
    require(final["augmented_images"] == 212, "The delivery is for a different augmentation")
    require(not final["weak_prior_quality_gain_accepted"], "The weak-prior evidence boundary changed")
    require(final["native_GS_projection_status"].startswith("known_shared"), "Projection limitation is missing")
    for row in final["render_metrics"]:
        require(row["all_H_count"] == 24 and row["rendered_H_count"] == 24, "H denominator changed")
        for domain in ["full_image", "fixed_target_rectangle"]:
            require(f'{row[domain]["mean_PSNR_dB"]:.4f}' in report, "Report PSNR does not match the metric")
            require(f'{row[domain]["mean_SSIM_INRIA"]:.6f}' in report, "Report SSIM does not match the metric")
    for row in final["downstream_cases"]:
        state = row["training"]
        require(state["iteration"] == 10000 and state["completed_backward_calls"] == 10000, "Incomplete training")
        require(f'{state["gaussian_count"]:,}' in report, "Report Gaussian count is incorrect")
        require(f'{row["filtered_points"]:,}' in report, "Report point count is incorrect")
        require(f'{state["elapsed_seconds"]:.3f}' in report, "Report training runtime is incorrect")
    same = read_json(run / "02_same_180_ba/COMMON_OBSERVATION_METRICS.json")
    for row in same["records"]:
        require(f'{row["reprojection_px"]["p95"]:.9f}' in report, "Report BA P95 is incorrect")
        require(row["reprojection_px"]["count"] == 336377, "Common observation denominator changed")
        receipt = read_json(run / "02_same_180_ba" / row["case"] / "RECEIPT.json")
        for phase in ["warmup", "joint"]:
            require(receipt[phase]["usable"] and receipt[phase]["termination_type"].endswith("NO_CONVERGENCE"),
                    "The documented real-data solver termination changed")
    require("NO_CONVERGENCE" in report and "7.322" in report, "Material limitations are missing from the report")
    browser = read_json(run / "viewer/BROWSER_VALIDATION.json")
    display = read_json(run / "viewer/FINAL_DISPLAY_VALIDATION.json")
    require(browser["status"] == "passed" and display["final_default_check_result"]["status"] == "passed",
            "Browser validation is incomplete")
    for case, expected_index in display["default_indices"].items():
        directory = run / "viewer/cases" / case
        manifest = read_json(directory / "display_manifest.json")
        path = read_json(directory / "camera_path.json")
        require(manifest["default_view"] == expected_index, "Viewer default index is incorrect")
        require(path[expected_index]["image_name"] == display["common_default_capture"], "Viewer default capture differs")
    for relative in display["visual_review_artifacts"]:
        require((run / "viewer" / relative).resolve().is_file(), f"Missing visual review artifact: {relative}")
    with urllib.request.urlopen("http://127.0.0.1:8195/SERVER_STATE.json", timeout=3) as response:
        server = json.load(response)
    require(Path(server["root"]).resolve() == run / "viewer" and server["status"] == "serving", "Wrong viewer service")
    for path in code.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for path in (run / "code_snapshot").iterdir():
        require((code / path.name).read_bytes() == path.read_bytes(), f"Numerical source changed after snapshot: {path.name}")
    snapshot = run / "delivery_snapshot"
    snapshot.mkdir()
    for path in [code / "README.md", Path(__file__).resolve(), run / "START_VIEWER.ps1"]:
        shutil.copy2(path, snapshot / path.name)
    examples = re.findall(r"```powershell\n(.*?)\n```", content["README.md"], re.DOTALL)
    require(len(examples) == 3, "Reproduction examples are incomplete")
    (snapshot / "powershell_examples.ps1").write_text("\n\n".join(examples), encoding="utf-8")
    result = {
        "status": "passed_native_text_and_recorded_evidence_consistency",
        "documents_read_as_native_text": [str(path) for path in documents],
        "local_links_checked": len(linked),
        "local_link_targets": linked,
        "numerical_metrics_checked_against_JSON": True,
        "common_observations": 336377,
        "all_H_images_per_case": 24,
        "real_data_BA_termination": "NO_CONVERGENCE_usable",
        "numerical_source_snapshot_unchanged": True,
        "python_sources_compiled": True,
        "PowerShell_examples_written_for_separate_parser_check": str(snapshot / "powershell_examples.ps1"),
        "viewer_service": server,
        "review_kind": "self_review",
        "document_hash_or_screenshot_used_as_acceptance": False,
        "known_projection_and_accuracy_limits_preserved": True,
        "timestamp_unix": time.time()
    }
    write_json(run / "DELIVERY_REVIEW.json", result)
    print({"status": result["status"], "local_links_checked": len(linked), "viewer_pid": server["pid"]})


if __name__ == "__main__":
    configure_stdout()
    main()

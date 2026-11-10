"""Check report denominators and real complete outputs without promoting failed branches."""

import argparse
from pathlib import Path

import numpy as np
import psutil

from atomic_json_windows import atomic_json, read_json


def gaussian_file(path, expected):
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 16384:
                raise ValueError("Invalid complete Gaussian PLY header")
            header.extend(line)
    lines = header.decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("Unexpected native Gaussian PLY format")
    count = int(next(line.split()[2] for line in lines if line.startswith("element vertex ")))
    properties = [line for line in lines if line.startswith("property ")]
    if count != expected or len(properties) != 62 or any(not line.startswith("property float ") for line in properties):
        raise ValueError("Native population or all62 Gaussian fields changed")
    if path.stat().st_size != len(header) + count * 62 * 4:
        raise ValueError("The full native Gaussian file has a missing or extra payload")
    values = np.memmap(path, dtype="<f4", mode="r", offset=len(header), shape=(count, 62))
    for start in range(0, count, 16384):
        if not np.isfinite(values[start:start+16384]).all():
            raise ValueError("A complete native Gaussian field is nonfinite")
    del values
    return {"vertex_count": count, "fields": 62, "full_payload_size_correct": True,
        "all_native_fields_finite": True, "geometry_accuracy_not_inferred": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    args.output.resolve().relative_to(args.campaign.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    comparison, modules = read_json(args.comparison), read_json(args.modules)
    cases = read_json(args.campaign / "CASES.json")["cases"]
    if {row["case"] for row in comparison["cases"]} != {row["case"] for row in cases} or len(cases) != 4:
        raise ValueError("The four raw recording denominator differs")
    if len(modules["modules"]) != 75 or len({row["id"] for row in modules["modules"]}) != 75 or modules["all75_effective"]:
        raise ValueError("The unique mechanism denominator or claim boundary differs")
    if sum(modules["status_counts"].values()) != 75:
        raise ValueError("The mechanism status counts disagree")
    result = {"status": "checking_current_outputs", "self_review_not_independent_review": True,
        "case_count": 4, "unique_modules": 75, "sides": [], "missing_and_failed_outputs_preserved": True}
    try:
        for module in modules["modules"]:
            for evidence in module["evidence"]:
                values = evidence.get("outcome_fields") or {}
                if values.get("complete_pair_oracle_passed") is False and values.get("observed_reduction_fraction") is not None:
                    raise ValueError("A failed short localization run was incorrectly promoted as speed")
        if args.final:
            required = [args.campaign / "CAMPAIGN_STATE.json", args.campaign / "EVALUATION_CAMPAIGN_STATE.json",
                args.campaign / "box2/localization_modules/STATE.json",
                args.campaign / "box2/localization_modules_compatibility_001/STATE.json",
                args.campaign / "box2/localization_process_receipt_retry_001/STATE.json",
                args.campaign / "box2/matching_isolated_schedule_001/STATE.json"]
            if any(read_json(path)["status"] in ("running", "waiting_for_primary_if_requested",
                    "waiting_for_owned_primary_and_localization", "running_isolated_complete_matching",
                    "waiting_for_current_compatibility_outcomes", "running_complete_process_receipt_retries") for path in required):
                raise RuntimeError("A required current audit has not resolved")
            if any("current_audit_still_running" in row["current_outcome_statuses"] for row in modules["modules"]):
                raise ValueError("The final mechanism ledger still contains an active audit")
            result["required_current_audits_terminal"] = True
        for case in cases:
            case_name = case["case"]
            equivalence = read_json(args.campaign / case_name / "INITIAL_COMPARISON.json")
            if not all(equivalence["checks"].values()):
                raise ValueError("A full raw initial equivalence check failed")
            for mode in ("baseline", "optimized"):
                side = args.campaign / case_name / mode
                state = read_json(side / "PIPELINE_STATE.json")
                reported_case = next(v for v in comparison["cases"] if v["case"] == case_name)
                reported_side = reported_case["sides"][mode]
                if reported_side["pipeline"]["status"] != state["status"]:
                    raise ValueError("The comparison snapshot is stale for the current side outcome")
                row = {"case": case_name, "mode": mode, "pipeline_status": state["status"],
                    "failure_not_promoted": state["status"].startswith("failed"),
                    "initial_complete_raw_equivalence": True}
                try:
                    row["exact_side_controller_alive"] = psutil.Process(state["pid"]).create_time() == state["process_birth"]
                except psutil.NoSuchProcess:
                    row["exact_side_controller_alive"] = False
                if state["status"] == "running" and not row["exact_side_controller_alive"]:
                    raise ValueError("A stale running side has no exact live controller")
                train_path = side / "training/STATE.json"
                if train_path.is_file():
                    train = read_json(train_path)
                    if train["status"] == "completed":
                        process = read_json(side / "training_process.json")
                        admission = read_json(side / "export/TRAINING_ADMISSION.json")
                        if (process["status"] != "completed" or process.get("returncode") != 0
                                or admission["status"] != "export_and_admission_verified_training_pending"
                                or train["iteration"] != train["target_iterations"]
                                or train["target_iterations"] != (10000 if mode == "baseline" else 12000)
                                or train["image_count"] != admission["training_count"]):
                            raise ValueError("Actual complete training and admission records disagree")
                        model_path = Path(train["final_model"])
                        if model_path.as_posix().startswith("/mnt/c/"):
                            model_path = Path("C:/" + model_path.as_posix()[7:])
                        row["complete_Gaussian_file"] = gaussian_file(model_path, train["gaussian_count"])
                evaluation_path = side / "evaluation/RESULT.json"
                if evaluation_path.is_file():
                    evaluation = read_json(evaluation_path)
                    frozen = read_json(side / "initial/EVALUATION_FRAMES.json")["rows"]
                    if len(evaluation["rows"]) != 24 or [v["image_name"] for v in frozen] != [v["image_name"] for v in evaluation["rows"]]:
                        raise ValueError("A failed frozen evaluation view was removed or reordered")
                    localized = sum(v["localized"] for v in evaluation["rows"])
                    if evaluation["summary"]["full_image"]["rendered_count"] != localized:
                        raise ValueError("The actual successful render count differs")
                    for view in evaluation["rows"]:
                        rendered = side / "evaluation" / (Path(view["image_name"]).stem + ".png")
                        if rendered.is_file() != bool(view["localized"]):
                            raise ValueError("The rendered image presence differs from actual evaluation outcomes")
                    row["complete_evaluation_denominator"] = 24
                    row["actual_rendered_count"] = localized
                    row["partial_quality_not_promoted"] = localized != 24
                result["sides"].append(row)
        complete_count = sum("complete_Gaussian_file" in row for row in result["sides"])
        if complete_count != comparison["completed_models"]:
            raise ValueError("The reported complete-model count differs from full payload validation")
        result["complete_model_count"] = complete_count
        result.update(status="current_receipt_and_complete_output_checks_passed_scope_limited",
            all_eight_side_controllers_finished=not any(v["exact_side_controller_alive"] for v in result["sides"]),
            GT_or_absolute_scale_accuracy_established=False, all75_effective=False)
    except BaseException as error:
        result.update(status="failed_current_delivery_check", error=repr(error))
        atomic_json(args.output / "RESULT.json", result)
        raise
    atomic_json(args.output / "RESULT.json", result)
    import json
    print(json.dumps({"status": result["status"], "sides": len(result["sides"]),
        "all_eight_side_controllers_finished": result["all_eight_side_controllers_finished"]}))


if __name__ == "__main__":
    main()

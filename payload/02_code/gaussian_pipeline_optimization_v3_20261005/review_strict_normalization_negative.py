"""Independently recheck the saved complete negative pair and actual owned exits."""

import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nn001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows(path):
    return Path(path.replace("/mnt/c/", "C:/"))


def equal_identity(row):
    current = data_identity(windows(row["path"]))
    return all(current[key] == row[key] for key in ("bytes", "sha256"))


def main():
    target = OUT / "NEGATIVE_SELF_REVIEW_001.json"
    if target.exists():
        raise FileExistsError("A negative review must retain its original receipt")
    state, generic = read(OUT / "c0/RESULT.json"), read(OUT / "g0/RESULT.json")
    left, right = (read(OUT / "c0/t0" / mode / "RESULT.json") for mode in ("native", "arithmetic"))
    checks = {
        "first_pair_stopped_by_registered_complete_cost_condition": state["status"] == "closed_negative_complete_arithmetic_screen0"
            and len(state["pairs"]) == 1 and state["net_complete_reduction_upper_bound_not_final_acceptance"] < .02,
        "all_generic_conditions_passed_without_acceptance": len(generic["checks"]) == 11 and all(generic["checks"].values()) and generic["new_optimization_count"] == 0,
        "all15_original_scientific_fields_and1016_rows_exact": len(left["scientific"]) == 15 and left["scientific"] == right["scientific"]
            and len(left["scientific"]["rows"]) == 1016,
        "all_original_generated_asset_bytes_exact": set(left["generated_assets"]) == {"input.ply", "cameras.json"}
            and all(windows(left["generated_assets"][name]["path"]).read_bytes() == windows(right["generated_assets"][name]["path"]).read_bytes() for name in left["generated_assets"]),
        "complete_source_input_and_library_identities_current": all(equal_identity(row) for row in state["sources"] + generic["sources"] + [generic["library"]] + left["sources"] + left["inputs"]),
        "both_complete_original_inputs_and_runtime_identical": len(left["inputs"]) == 1036 and len(left["sources"]) == 17
            and left["inputs"] == right["inputs"] and left["sources"] == right["sources"] and left["runtime"] == right["runtime"] and left["runtime_config"] == right["runtime_config"],
        "all6_saved_full_comparison_conditions_passed": len(state["pairs"][0]["comparison"]["checks"]) == 6 and all(state["pairs"][0]["comparison"]["checks"].values()),
        "old_failures_and_complete_family_fees_retained": all(state["known_extra_costs"][name] > 0 for name in (
            "exact_normalization_oracle_001", "exact_normalization_oracle_002", "exact_normalization_scene_001")),
        "no_other_fresh_pair_silently_executed": not (OUT / "c0/t1").exists() and not (OUT / "c0/t2").exists(),
    }
    win_ids, linux_ids = [], []
    for prefix in ("g0", "c0_first"):
        lease, guard = read(OUT / (prefix + "_lease.json")), read(OUT / (prefix + "_GUARD.json"))
        checks[prefix + "_actual_guard_and_lease_closed0"] = lease["returncode"] == guard["returncode"] == 0 and bool(lease["lease_released_utc"])
        win_ids.extend(guard["owned_process_identities"])
        win_ids.extend({"pid": lease[key + "pid"], "creation_time": lease[key + "creation_time"]} for key in ("guard_", "child_"))
    linux_ids.append(read(OUT / "g0/WSL_RESOURCE.json"))
    for mode in ("native", "arithmetic"):
        leaf, node = OUT / "c0/t0" / mode, OUT / "c0/t0" / (mode + "_node")
        controller = read(node / "NODE.json")
        trace, identity = read(leaf / "NORMALIZATION_TRACE.json"), read(leaf / "IDENTITY_TRACE.json")
        checks[mode + "_full_adapter_identity_and_resources_passed"] = all(read(leaf / "NORMALIZATION_ADAPTER_CHECKS.json")["checks"].values())
        checks[mode + "_original_route_restored_and_all1016_called"] = trace["trace"]["original_binding_restored"] and trace["trace"]["calls"] == 1016 and trace["exception_type"] is None
        checks[mode + "_four_complete_identity_phases"] = all(identity["checks"].values()) and [phase["returned"] for phase in identity["phases"]] == [17, 1036, 1036, 17]
        checks[mode + "_actual_owner_drained_and_guard_exited0"] = controller["status"] == "completed" and controller["returncode"] == 0
        win_ids.extend(controller["guard"]["owned_process_identities"])
        win_ids.append(controller["owner"])
        win_ids.append(state["pairs"][0]["rows"][mode]["owner"])
        linux_ids.append(read(leaf / "WSL_RESOURCE.json"))
    win_ids = list({(item["pid"], item["creation_time"]): item for item in win_ids}.values())
    alive = []
    for item in win_ids:
        try:
            process = psutil.Process(item["pid"])
            if process.create_time() == item["creation_time"]:
                alive.append(item)
        except psutil.NoSuchProcess:
            pass
    payload = [{"pid": item["pid"], "creation_time": item["creation_time"]} for item in linux_ids]
    code = "import json,psutil; rows=" + repr(payload) + "; answer=[]\n"
    code += "for i in rows:\n row=dict(i,alive=False)\n try:\n  p=psutil.Process(i['pid']); row['alive']=p.create_time()==i['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n answer.append(row)\nprint(json.dumps(answer))"
    linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["all_saved_Windows_and_Linux_PID_birth_instances_ended"] = not alive and not any(item["alive"] for item in linux)
    report = (DOC / "R97_V2_ARITHMETIC_RESULTS.md").read_text(encoding="utf-8")
    task = (DOC / "R97_V2_ARITHMETIC_TASK.md").read_text(encoding="utf-8")
    links = [windows(path) for path in re.findall(r"\]\((C:/[^)]+)\)", report)]
    expected_pending = OUT / "FINAL_POST_EXIT_001.json"
    checks["report_original_text_claims_and_references_match_negative_scope"] = all(part in report for part in ("33/50", "1.814921%", "0.858510%", "不执行第2/3对", "未验证", "自审"))
    checks["all_report_targets_existing_or_explicit_final_pending"] = bool(links) and all(path.is_file() or path == expected_pending or path == target for path in links)
    checks["task_retains_preregistered_gate_and_original_quality"] = all(part in task for part in ("净>=2%", "六侧", "两个Scene生成文件原字节", "第1对", "原17sources/1036input"))
    parent = read(DOC / "CURRENT_STATE.json")
    accepted = parent["accepted_mechanisms"]
    checks["parent33_unique_and_no_R97_counted"] = parent["additional_accepted_count"] == 33 and len(accepted) == len(set(accepted)) == 33 and not any("R97" in name for name in accepted)
    atomic_json(target, {"status": "verified_negative_complete_first_pair" if all(checks.values()) else "failed_verification",
        "checks": checks, "Windows_saved_instances": win_ids, "Linux_actual_current_checks": linux,
        "alive_Windows": alive, "source": data_identity(Path(__file__)), "reviewer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "original_report_read_as_text": True, "independent_agent_review": False,
        "new_optimization_count": 0, "parent_accepted_count": 33,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"checks": checks, "Windows_instances": len(win_ids), "Linux_instances": len(linux)}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The separate negative self-review failed")


if __name__ == "__main__":
    main()

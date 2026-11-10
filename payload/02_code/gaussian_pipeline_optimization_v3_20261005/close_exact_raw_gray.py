"""Close R147 sources, real publisher instances, native text and known cost ledger."""

import json
import msvcrt
import os
from pathlib import Path
import re
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json,data_identity
from review_exact_raw_gray import cost_snapshot


def reject_duplicate_keys(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError("A final native JSON contains a duplicate key: "+key)
        result[key]=value
    return result


def load(path):
    return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=reject_duplicate_keys)


def main():
    started=time.perf_counter()
    base=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001"
    target=base/"FINAL_COST_001.json"
    if target.exists():
        raise FileExistsError("The final source/text/cost snapshot must have a new path")
    checks={}
    saved=set()
    leases=sorted(base.glob("LEASE_*.json"))
    for path in leases:
        row=load(path)
        if row.get("returncode")!=0 or not row.get("lease_released_utc"):
            raise ValueError("A measured R147 task has not completed and released")
        for prefix in ("guard","child"):
            saved.add((row[prefix+"_pid"],row[prefix+"_creation_time"]))
    for pid,born in saved:
        try:
            actual=psutil.Process(pid)
            if actual.create_time()==born:
                raise ValueError("A completed exact instance still owns a live process")
        except psutil.NoSuchProcess:
            pass
    checks["all_ten_real_lease_owners_review_and_publisher_instances_exited"]=len(leases)==10
    shared=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    owner=load(shared/"LOCK_OWNER.json")
    if owner.get("status")!="released":
        raise ValueError("The resource owner has not recorded its released state")
    with (shared/"exclusive_benchmark.lock").open("r+b",buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
    checks["actual_shared_OS_lock_available_and_owner_released"]=True
    current_source_checks=0
    actual_snapshot_checks=0
    for name in ("p0","b0","g0","s0","g1","e0","c0"):
        receipt=load(base/name/("PROTOCOL.json" if name=="c0" else "RESULT.json"))
        for position,source in enumerate(receipt.get("sources",[])):
            path=Path(source["path"])
            if path.suffix in (".py",".cpp",".dll"):
                if data_identity(path)!=source:
                    raise ValueError("A current actually executed scientific source changed")
                current_source_checks+=1
            if path.suffix in (".py",".cpp"):
                frozen=base/name/"source_snapshot"/(str(position)+"_"+path.name)
                if name=="b0":
                    frozen=base/name/"source_snapshot"/path.name
                if frozen.read_bytes()!=path.read_bytes():
                    raise ValueError("An actually executed text source snapshot differs from its immutable source")
                actual_snapshot_checks+=1
    checks["all_component_and_complete_scientific_sources_and_actual_text_snapshot_bytes_verified"]=current_source_checks>=70 and actual_snapshot_checks>=50
    doc=ROOT/"research/2026-10-05-gaussian-optimization-round3-50"
    parent=load(doc/"CURRENT_STATE.json")
    acceptance=load(base/"ACCEPTANCE.json")
    mechanism="R147_EXACT_RGGB_QUALITY_INPUT_FUSION"
    checks["parent25_unique_accepted_mechanisms_and_receipts"]=parent["additional_accepted_count"]==len(parent["accepted_mechanisms"])==len(
        set(parent["accepted_mechanisms"]))==len(parent["acceptance_receipts"])==25 and parent["accepted_mechanisms"][-1]==mechanism and all(
        (ROOT/path).is_file() for path in parent["acceptance_receipts"])
    checks["only_one_R147_item_and_related_original_mechanisms_not_recounted"]=parent["R147_new_optimization_count"]==acceptance["new_optimization_count"]==1 and acceptance[
        "related_R104_R141_composition_is_not_a_second_item"]
    bodies={name:(doc/name).read_text(encoding="utf-8") for name in ("R147_EXACT_RAW_GRAY_RESULTS.md","CANDIDATE_BOARD.md","PLAN.md",
        "OPTIMIZE_CHECKLIST.md","GENERALIZATION_PROTOCOL.md","NEXT_RAW_GRAY_INPUT_BRIEF.md","NEXT_QUALITY_CORNER_COST_BRIEF.md")}
    report=bodies["R147_EXACT_RAW_GRAY_RESULTS.md"]
    checks["native_full_report_scope_numbers_fees_negative_and_unrun_limits"]=all(value in report for value in (
        "25/50","7338","1002","11583","4,202,938","49.7659","41.1342","56.7809","48.2558","固定6侧","原生",
        "未运行新训练","公开","不是实测Oxford RAW","不代表研究文字工时"))
    checks["current_native_documents_all_preserve25_and_single_chat_and_pending_new_objects"]=all("25/50" in text[:2000] for text in bodies.values()) and not parent[
        "active_candidates"] and "single_chat_no_subagents" in parent["next_route"] and acceptance["generalization_pending"]
    local_links=[Path(match) for text in bodies.values() for match in re.findall(r"\]\((C:/[^)]+)\)",text)]
    checks["all_native_report_local_links_existing_or_the_receipt_being_created"]=bool(local_links) and all(path.is_file() or path==target for path in local_links)
    complete=load(base/"c0/RESULT.json")
    cost=cost_snapshot(base,complete,time.perf_counter()-started)
    checks["all_known_post_reviewer_publisher_close_fees_phase5percent_and_whole_limits"]=cost["net_reductions"]["phase_seconds"]>=.05 and cost[
        "whole_median_ratio"]<=1.01 and all(pair["whole_seconds"]["fused"]<=1.1*pair["whole_seconds"]["native"] for pair in complete["pairs"])
    if not all(checks.values()):
        raise ValueError("A post-exit exact source, text, parent or complete cost closure failed: "+repr(checks))
    process=psutil.Process()
    result={"status":"accepted_R147_post_exit_sources_native_text_parent25_and_all_known_fees_verified",
        "checks":checks,"real_completed_leases":len(leases),"actual_saved_owner_instances_exited":len(saved),
        "current_scientific_source_checks":current_source_checks,"actual_executed_text_snapshot_checks":actual_snapshot_checks,
        "native_text_documents_directly_checked":list(bodies),"existing_local_links_checked":len(local_links),"cost":cost,
        "new_optimization_count":1,"parent_additional_accepted_count":25,"parent_native_goal_remains_active_and_incomplete":True,
        "closer_PID":process.pid,"closer_creation_time":process.create_time(),"node_before_last_serialization_seconds":time.perf_counter()-started,
        "last_JSON_stdout_and_exit_tail_not_independently_metered_or_assumed_zero":True,"source":data_identity(Path(__file__))}
    atomic_json(target,result)
    if not all(path.is_file() for path in local_links):
        raise ValueError("A final native report link failed after receipt creation")
    print({"status":result["status"],"checks":len(checks),"stage_reduction":cost["net_reductions"]["phase_seconds"],
           "whole_reduction":cost["net_reductions"]["whole_seconds"],"known_shared_seconds":cost["known_extra_fee_seconds"],"new_optimization_count":1},flush=True)


if __name__=="__main__":
    main()

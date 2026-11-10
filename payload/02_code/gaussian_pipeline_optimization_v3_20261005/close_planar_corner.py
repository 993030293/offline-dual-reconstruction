"""Verify actual exit, current native text, unique parent count, and final known costs."""

import json
import msvcrt
from pathlib import Path
import re
import time

import psutil

import profile_quality_corner_cost as reference
from review_planar_corner import cost_snapshot


def reject_duplicates(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError("A final native JSON contains a duplicate key: "+key)
        result[key]=value
    return result


def load(path):
    return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=reject_duplicates)


def main():
    started=time.perf_counter()
    root=reference.ROOT
    base=root/"04_outputs/gaussian_pipeline_optimization_v3_20261005/qc001"
    doc=root/"research/2026-10-05-gaussian-optimization-round3-50"
    target=base/"FINAL_COST_001.json"
    if target.exists():
        raise FileExistsError("The final corner source/text/cost snapshot must have a new path")
    process=psutil.Process()
    reference.atomic_json(base/"FINAL_CLOSER_PROCESS.json",{"pid":process.pid,"creation_time":process.create_time()})
    checks={}
    saved=set()
    leases=sorted(base.glob("LEASE_*.json"))
    for path in leases:
        row=load(path)
        if row.get("returncode")!=0 or not row.get("lease_released_utc"):
            raise ValueError("A measured original corner owner has not completed and released")
        for prefix in ("guard","child"):
            saved.add((row[prefix+"_pid"],row[prefix+"_creation_time"]))
    for pid,born in saved:
        try:
            if psutil.Process(pid).create_time()==born:
                raise ValueError("A completed saved exact process instance is still alive")
        except psutil.NoSuchProcess:
            pass
    checks["all_ten_actual_lease_owners_reviewer_and_publisher_instances_exited"]=len(leases)==10
    shared=root/"04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    if load(shared/"LOCK_OWNER.json").get("status")!="released":
        raise ValueError("The original resource owner has not recorded release")
    with (shared/"exclusive_benchmark.lock").open("r+b",buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
    checks["actual_shared_OS_lock_available_and_recorded_owner_released"]=True
    reviewer=load(base/"r0/RESULT.json")
    for identity in reviewer["sources_verified"]+[reviewer["source"],load(base/"PUBLICATION_001.json")["source"]]:
        if reference.data_identity(Path(identity["path"]))!=identity:
            raise ValueError("An actually executed source, native library or publisher changed")
    for row in reviewer["snapshots_verified"]:
        if Path(row["snapshot"]).read_bytes()!=Path(row["original"]).read_bytes():
            raise ValueError("A frozen actual code-text snapshot changed")
    checks["all97_current_executed_source_identities_and84_code_snapshots_exact"]=len(reviewer["sources_verified"])+2==97 and len(reviewer["snapshots_verified"])==84
    parent=load(doc/"CURRENT_STATE.json")
    accepted=load(base/"ACCEPTANCE.json")
    mechanism="R149_EXACT_PLANAR_CORNER_RESPONSE_CONSUMER"
    checks["parent26_unique_accepted_mechanisms_and_existing_receipts"]=parent["additional_accepted_count"]==len(parent["accepted_mechanisms"])==len(set(parent["accepted_mechanisms"]))==len(parent["acceptance_receipts"])==26 and parent["accepted_mechanisms"][-1]==mechanism and all((root/path).is_file() for path in parent["acceptance_receipts"])
    checks["R149_one_backend_only_R147_composition_R148_and_prefix_not_recounted"]=parent["R149_new_optimization_count"]==accepted["new_optimization_count"]==1 and parent["R148_new_optimization_count"]==0 and accepted["R147_R141_related_composition_not_a_second_item"] and accepted["one_layout_and_native_selection_backend_not_individually_ablated_or_counted_twice"]
    filenames=("R149_PLANAR_CORNER_RESULTS.md","NEXT_POST_CORNER_COST_BRIEF.md","CANDIDATE_BOARD.md","PLAN.md",
        "OPTIMIZE_CHECKLIST.md","GENERALIZATION_PROTOCOL.md","NEXT_QUALITY_CORNER_COST_BRIEF.md")
    bodies={name:(doc/name).read_text(encoding="utf-8") for name in filenames}
    report=bodies["R149_PLANAR_CORNER_RESULTS.md"]
    checks["direct_report_text_matches_actual_counts_times_fees_memory_negative_and_unverified_scope"]=all(value in report for value in (
        "26/50 active","7,338","1,002","11,583","42.605882","34.718218","49.926516","42.076247",
        "14.4%","12.7%","225.074219","253.054688","29项","17项","0.320868%","generalization_pending",
        "不是盲测","没有新高斯训练","最后尾部","分离自审不是独立agent审查","未单独消融"))
    checks["all_seven_changed_control_texts_latest26_and_historical_scope_preserved"]=all("26/50" in body[:1400] for body in bodies.values())
    links=[]
    for name,body in bodies.items():
        for match in re.finditer(r"\]\((C:/[^)]+)\)",body):
            path=Path(match.group(1))
            if not path.is_file() and path!=target:
                raise FileNotFoundError("A native control-text local link is unavailable: "+str(path))
            links.append({"document":name,"target":str(path),"declared_final_receipt":path==target})
    checks["all_direct_native_local_links_exist_except_this_declared_final_output"]=len(links)>10
    complete=load(base/"c0/RESULT.json")
    cost=cost_snapshot(base,complete,time.perf_counter()-started)
    checks["all_known_final_phase_reduction_at_least5percent"]=cost["net_reductions"]["phase_seconds"]>=.05
    checks["all_known_final_whole_median_at_most1point01_and_each_within1point1"]=cost["whole_median_ratio"]<=1.01 and all(pair["whole_seconds"]["planar"]<=1.1*pair["whole_seconds"]["current"] for pair in complete["pairs"])
    if not all(checks.values()):
        raise ValueError("A final original text, lineage, parent count, exit or cost check failed")
    reference.atomic_json(target,{"status":"closed_accepted_one_planar_corner_backend_parent26_active_with_final_known_costs",
        "checks":checks,"cost":cost,"actual_leases":len(leases),"exact_prior_lease_instances_exited":len(saved),
        "current_executed_source_identity_checks":97,"actual_code_text_snapshot_checks":84,
        "native_control_texts_read":list(bodies),"local_links_checked":links,
        "batch_text_patch_failure_preserved":"PLAN header context mismatch caused no write; actual text reread and bounded patch retried; no science change or count",
        "self_review_not_independent_agent":True,"prior_goal_still_active":True,"new_optimization_count":1,
        "known_final_fee_includes_closer_before_last_JSON_stdout_exit_tail":True,
        "last_JSON_stdout_exit_tail_not_separately_metered_or_assumed_zero":True,"source":reference.data_identity(Path(__file__)),
        "node_seconds_before_last_write":time.perf_counter()-started})
    print({"status":"closed_accepted_parent26_active","checks":len(checks),"known_fee_seconds":cost["known_extra_fee_seconds"],
        "stage_net_reduction":cost["net_reductions"]["phase_seconds"],"whole_net_reduction":cost["net_reductions"]["whole_seconds"]},flush=True)


if __name__=="__main__":
    main()

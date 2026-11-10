"""Recheck full RAW input contracts and costs in a separate root self-review."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
V1=ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2=ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0]=[str(V1),str(V2)]
from runtime import atomic_json,data_identity
from raw_initial_invariants import compare_initial_outputs


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cost_snapshot(base,complete,additional_seconds=0.):
    leases=[]
    for path in sorted(base.glob("LEASE_*.json")):
        row=load(path)
        if "child_elapsed_seconds" in row:
            leases.append({"path":str(path),"seconds":row["child_elapsed_seconds"],"returncode":row["returncode"]})
    leaf=sum(row["whole_seconds"] for row in complete["runs"])
    comparison=sum(pair["comparison_seconds_charged_each_side"] for pair in complete["pairs"])
    all_lease=sum(row["seconds"] for row in leases)
    shared=all_lease-leaf-comparison+additional_seconds
    if shared<0:
        raise ValueError("The actual shared fee ledger became negative")
    medians={name:{mode:statistics.median(pair[name][mode] for pair in complete["pairs"])
                  for mode in ("native","fused")} for name in ("phase_seconds","whole_seconds")}
    charged={name:{mode:value+shared/6 for mode,value in modes.items()} for name,modes in medians.items()}
    reduction={name:1-modes["fused"]/modes["native"] for name,modes in charged.items()}
    return {"completed_lease_envelopes":leases,"all_completed_lease_seconds":all_lease,
            "actual_formal_leaf_seconds_already_in_side_windows":leaf,
            "actual_comparison_once_removed_from_shared_and_charged_each_side":comparison,
            "known_extra_fee_seconds":shared,"fixed_fee_amortization_sides":6,
            "raw_complete_medians":medians,"after_all_known_fees_complete_medians":charged,"net_reductions":reduction,
            "whole_median_ratio":charged["whole_seconds"]["fused"]/charged["whole_seconds"]["native"],
            "unmetered_model_research_text_work_not_assumed_zero_or_called_processing_time":True,
            "last_JSON_stdout_and_exit_tail_not_independently_metered_at_this_snapshot":True}


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    base=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001"
    output=args.output.resolve()
    output.relative_to(base)
    output.mkdir(parents=True,exist_ok=False)
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time(),"scope":"Windows"})
    checks={}
    results={name:load(base/name/"RESULT.json") for name in ("p0","b0","g0","s0","g1","c0","e0")}
    state={"status":"running","review_kind":"separate_root_self_review_not_independent_agent","accepted_count_contribution":0}
    try:
        checks["native_direct_failure_preserved_not_used_as_reference"]=not results["p0"]["native_direct_gray_exact"] and not results["p0"][
            "native_direct_real_six_fields_exact"] and (base/"p0/FIRST_GRAY_FAILURE.npz").is_file()
        checks["all_exhaustive_and_generic_contracts_passed"]=results["g0"]["coefficient_triplets"]==2**24 and all(results["g0"]["checks"].values())
        checks["all_fixed_RAW128_gray_six_fields_and_cache_contracts_passed"]=results["s0"]["all128_complete_gray_arrays_bits_exact"] and len(results["s0"][
            "pairs"])==3 and all(all(pair["checks"].values()) for pair in results["s0"]["pairs"])
        checks["both_real_initial_entry_errors_partial_cache_recovery_warm_and_corrupt_cache_passed"]=all(results["g1"]["checks"].values()) and len(results["g1"]["checks"])==12
        checks["public32_RGB_derived_operator_only_all_bits_fields_inputs_closed"]=results["e0"]["status"]=="public_RGB_derived_Bayer_operator_check_passed_not_RAW_GT_or_whole_generalization" and len(
            results["e0"]["rows"])==32 and all(results["e0"]["checks"].values())
        complete=results["c0"]
        checks["original_AB_BA_AB_three_complete_fresh_pairs"]=complete["status"]=="three_complete_fresh_pairs_finished_fees_recovery_review_pending" and len(
            complete["pairs"])==3 and [pair["order"] for pair in complete["pairs"]]==[["native","fused"],["fused","native"],["native","fused"]] and len(complete["runs"])==6
        reference=Path(load(base/"c0/PROTOCOL.json")["reference_initial"])
        scientific=[]
        for trial in range(3):
            folder=base/"c0"/("t"+str(trial))
            pair=complete["pairs"][trial]
            protection={"pair":compare_initial_outputs(folder/"native/initial",folder/"fused/initial"),
                "native_reference":compare_initial_outputs(reference,folder/"native/initial"),
                "fused_reference":compare_initial_outputs(reference,folder/"fused/initial")}
            left,right=folder/"native_quality",folder/"fused_quality"
            names=sorted(path.name for path in left.glob("*.json"))
            all_cache_equal=len(names)==7338 and names==sorted(path.name for path in right.glob("*.json")) and all(
                (left/name).read_bytes()==(right/name).read_bytes() for name in names)
            scientific.append({"trial":trial,"all7338_cache_files_exact":all_cache_equal,"original_invariants":protection})
            checks["t"+str(trial)+"_all_original7338_decisions_six_fields1002_JPEG_K_pose_policy_pairs_exact"]=all_cache_equal and all(
                value["all_protected"] for value in protection.values()) and all(pair["checks"].values())
        identities=[]
        snapshots=[]
        for name,result in results.items():
            for source in result.get("sources",[]):
                if Path(source["path"]).suffix in (".py",".cpp",".dll"):
                    if data_identity(Path(source["path"]))!=source:
                        raise ValueError("An executed native scientific source changed after verification")
                    identities.append(source)
            folder=base/name/"source_snapshot"
            if folder.is_dir():
                snapshots.extend(str(path) for path in folder.iterdir() if path.is_file())
        checks["all_executed_current_scientific_sources_and_DLL_unchanged"]=len(identities)>30 and len(snapshots)>30
        full_file_bytes=sum(path.stat().st_size for path in (base/"c0").rglob("*") if path.is_file())
        checks["actual_full_output_budget_below12GiB"]=full_file_bytes<=12*1024**3
        for name in ("p0","g0","s0","g1","e0"):
            close=load(base/name/"RESOURCE_CLOSE.json")
            required=[value for key,value in close.items() if key not in ("node_seconds","counted")]
            if not all(value is True for value in required):
                raise ValueError("An original complete component lifecycle failed to close")
        checks["all_component_original_threads_planes_builders_and_DLL_closed"]=True
        saved={}
        for path in base.glob("LEASE_*.json"):
            row=load(path)
            if path.name=="LEASE_R0.json":
                continue
            if row.get("returncode")!=0 or not row.get("lease_released_utc"):
                raise ValueError("A prerequisite real task did not exit and release")
            for prefix in ("guard","child"):
                saved[(row[prefix+"_pid"],row[prefix+"_creation_time"])]=str(path)
        for path in list(base.glob("GUARD_*.json"))+list((base/"c0").glob("t*/*_PROCESS.json")):
            if path.name=="GUARD_R0.json":
                continue
            row=load(path)
            if row["status"]!="completed" or row["returncode"]!=0 or row["peak_tree_RSS_bytes"]>row["maximum_RSS_bytes"] or row[
                    "elapsed_seconds"]>row["maximum_seconds"]:
                raise ValueError("A real complete process guard failed")
            for value in row.get("owned_process_identities",[]):
                saved[(value["pid"],value["creation_time"])]=str(path)
        for pid,born in saved:
            try:
                actual=psutil.Process(pid)
                if actual.create_time()==born:
                    raise ValueError("A saved completed instance still owns a live process")
            except psutil.NoSuchProcess:
                pass
        checks["all_prior_saved_exact_PID_creation_time_instances_exited_and_resource_guards_passed"]=bool(saved)
        parent=load(ROOT/"research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
        checks["prior24_unique_mechanisms_receipts_remain_and_no_R147_premature_count"]=parent["additional_accepted_count"]==len(parent["accepted_mechanisms"])==len(set(
            parent["accepted_mechanisms"]))==len(parent["acceptance_receipts"])==24 and parent["R147_new_optimization_count"]==0 and all((ROOT/path).is_file() for path in parent["acceptance_receipts"])
        cost=cost_snapshot(base,complete,time.perf_counter()-started)
        checks["after_all_current_known_costs_complete_phase_at_least5percent"]=cost["net_reductions"]["phase_seconds"]>=.05
        checks["after_all_current_known_costs_whole_median_at_most1point01"]=cost["whole_median_ratio"]<=1.01
        checks["each_actual_complete_whole_within1point1"]=all(pair["whole_seconds"]["fused"]<=1.1*pair["whole_seconds"]["native"] for pair in complete["pairs"])
        if not all(checks.values()):
            raise ValueError("A full original source, semantic, resource or complete-cost review contract failed")
        state.update(status="separate_root_self_review_passed_post_exit_cost_and_text_publication_pending",checks=checks,scientific=scientific,
            current_scientific_source_identity_count=len(identities),executed_text_source_snapshot_count=len(snapshots),
            actual_saved_prior_instances_exited=len(saved),output_bytes=full_file_bytes,cost_before_reviewer_exit=cost,
            source=data_identity(Path(__file__)),sources_verified=identities,snapshots_verified=snapshots,
            node_seconds=time.perf_counter()-started,not_run=["new_training","mapper","LIO_replay","GT_metrics","unseen_object_whole_chain","production_preview_change"])
        atomic_json(output/"RESULT.json",state)
        print({"status":state["status"],"checks":len(checks),"stage_net_reduction":cost["net_reductions"]["phase_seconds"],
               "whole_net_reduction":cost["net_reductions"]["whole_seconds"],"known_shared_seconds":cost["known_extra_fee_seconds"],"counted":False},flush=True)
    except BaseException as error:
        state.update(status="failed",checks=checks,error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        atomic_json(output/"FAILURE.json",state)
        raise


if __name__=="__main__":
    main()

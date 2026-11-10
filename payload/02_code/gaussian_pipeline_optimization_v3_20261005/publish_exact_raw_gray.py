"""Publish scoped R147 acceptance only after real reviewer exit and source checks."""

import json
from pathlib import Path
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json,data_identity
from review_exact_raw_gray import cost_snapshot


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started=time.perf_counter()
    base=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001"
    if (base/"ACCEPTANCE.json").exists() or (base/"PUBLICATION_001.json").exists():
        raise FileExistsError("The scoped R147 acceptance must have a new path")
    reviewer=load(base/"r0/RESULT.json")
    if reviewer["status"]!="separate_root_self_review_passed_post_exit_cost_and_text_publication_pending" or not all(reviewer["checks"].values()):
        raise ValueError("The separated real full-input reviewer did not pass")
    lease=load(base/"LEASE_R0.json")
    if lease.get("returncode")!=0 or not lease.get("lease_released_utc"):
        raise ValueError("The actual reviewer did not exit and release")
    for prefix in ("guard","child"):
        try:
            process=psutil.Process(lease[prefix+"_pid"])
            if process.create_time()==lease[prefix+"_creation_time"]:
                raise ValueError("A saved reviewer controller still owns resources")
        except psutil.NoSuchProcess:
            pass
    protocol=load(base/"c0/PROTOCOL.json")
    if protocol["sources"]!=[data_identity(Path(source["path"])) for source in protocol["sources"]]:
        raise ValueError("An actual complete campaign or original dependency source changed")
    if reviewer["source"]!=data_identity(Path(reviewer["source"]["path"])):
        raise ValueError("The actually executed reviewer source changed")
    parent=load(ROOT/"research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if parent["additional_accepted_count"]!=24 or parent["R147_new_optimization_count"]!=0:
        raise ValueError("The preserved parent count changed before scoped acceptance")
    complete=load(base/"c0/RESULT.json")
    cost=cost_snapshot(base,complete,time.perf_counter()-started)
    if cost["net_reductions"]["phase_seconds"]<.05 or cost["whole_median_ratio"]>1.01 or not all(
            pair["whole_seconds"]["fused"]<=1.1*pair["whole_seconds"]["native"] for pair in complete["pairs"]):
        raise ValueError("The after-review complete cost or actual whole guard failed")
    receipt={"status":"accepted_scoped_exact_RAW_quality_input_fusion_under_existing_V2_V4",
        "mechanism":"R147_EXACT_RGGB_QUALITY_INPUT_FUSION","new_optimization_count":1,
        "local_effect_confirmed":True,"generalization_pending":True,
        "raw_frames":7338,"selected_frames":1002,"candidate_pairs":11583,"fresh_complete_pairs":3,
        "all_original_quality_fields_decisions_K_coarse_poses_JPEG_and_pairs_exact":True,
        "control":"accepted_Q78_R99_R104_R141_Q50_Q53_P18","same_family_first_native_direct_rounding_failure_preserved":str(base/"p0/EQUIVALENCE.json"),
        "generic":str(base/"g0/RESULT.json"),"component":str(base/"s0/RESULT.json"),"complete_pairs":str(base/"c0/RESULT.json"),
        "real_interruption_and_cache_recovery":str(base/"g1/RESULT.json"),"public_RGB_derived_operator_only":str(base/"e0/RESULT.json"),
        "review":str(base/"r0/RESULT.json"),"review_kind":reviewer["review_kind"],"cost_before_publisher_exit":cost,
        "source_support":"current_Windows_x64_AVX2_OpenCV4point13_uint8_RGGB_positive_row_stride_unit_column_stride_pixel_bound_original_fallback",
        "related_R104_R141_composition_is_not_a_second_item":True,
        "not_run":["new_training","new_SfM_or_mapper","full_LIO_replay","GT_metrics","unseen_object_whole_chain","production_or_preview_deployment"]}
    atomic_json(base/"ACCEPTANCE.json",receipt)
    atomic_json(base/"PUBLICATION_001.json",{"status":"scoped_acceptance_published_parent_and_text_update_pending","source":data_identity(Path(__file__)),
        "reviewer_exact_instances_exited":True,"all22_actual_complete_scientific_sources_unchanged":len(protocol["sources"])==22,
        "cost_before_this_publisher_exit":cost,"node_seconds":time.perf_counter()-started,"counted":False})
    print({"status":receipt["status"],"new_optimization_count":1,"complete_phase_reduction":cost["net_reductions"]["phase_seconds"],
           "complete_whole_reduction":cost["net_reductions"]["whole_seconds"]},flush=True)


if __name__=="__main__":
    main()

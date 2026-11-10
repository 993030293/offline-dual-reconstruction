"""Publish one scoped planar corner mechanism after actual review exit and cost checks."""

from pathlib import Path
import time

import psutil

import profile_quality_corner_cost as reference
from review_planar_corner import cost_snapshot


def main():
    started=time.perf_counter()
    root=reference.ROOT
    base=root/"04_outputs/gaussian_pipeline_optimization_v3_20261005/qc001"
    load,atomic_json,identity=reference.load,reference.atomic_json,reference.data_identity
    if (base/"ACCEPTANCE.json").exists() or (base/"PUBLICATION_001.json").exists():
        raise FileExistsError("The scoped planar acceptance requires a new receipt path")
    reviewer=load(base/"r0/RESULT.json")
    if reviewer["status"]!="separate_root_self_review_passed_post_exit_final_cost_and_text_publication_pending" or not all(reviewer["checks"].values()):
        raise ValueError("The separately executed root self-review did not pass")
    lease=load(base/"LEASE_R0.json")
    if lease.get("returncode")!=0 or not lease.get("lease_released_utc"):
        raise ValueError("The actual reviewer did not exit and release")
    for prefix in ("guard","child"):
        try:
            if psutil.Process(lease[prefix+"_pid"]).create_time()==lease[prefix+"_creation_time"]:
                raise ValueError("A saved reviewer instance still owns a live process")
        except psutil.NoSuchProcess:
            pass
    protocol=load(base/"c0/PROTOCOL.json")
    if protocol["sources"]!=[identity(Path(row["path"])) for row in protocol["sources"]] or reviewer["source"]!=identity(Path(reviewer["source"]["path"])):
        raise ValueError("An executed complete scientific or review source changed")
    parent=load(root/"research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if parent["additional_accepted_count"]!=25 or parent["R149_new_optimization_count"]!=0 or parent["R148_new_optimization_count"]!=0:
        raise ValueError("The protected current parent count changed before acceptance")
    complete=load(base/"c0/RESULT.json")
    cost=cost_snapshot(base,complete,time.perf_counter()-started)
    if cost["net_reductions"]["phase_seconds"]<.05 or cost["whole_median_ratio"]>1.01 or not all(pair["whole_seconds"]["planar"]<=1.1*pair["whole_seconds"]["current"] for pair in complete["pairs"]):
        raise ValueError("The complete after-review cost failed its existing frozen gate")
    receipt={"status":"accepted_scoped_exact_planar_corner_quality_consumer_under_existing_V2_V4",
        "mechanism":"R149_EXACT_PLANAR_CORNER_RESPONSE_CONSUMER","new_optimization_count":1,
        "parent_accepted_count_before":25,"local_effect_confirmed":True,"generalization_pending":True,
        "control":"accepted_R147_R141_R104_Q78_R99_Q50_Q53_P18","raw_frames":7338,"eligible_frames":6797,
        "selected_frames":1002,"candidate_pairs":11583,"fresh_complete_pairs":3,
        "all_original_six_fields_frame_decisions_K_coarse_metric_poses_JPEG_and_pairs_exact":True,
        "response_coordinate_order_operator_cases":{"generic":48,"actual_RAW":32,"public_RGB_derived":32},
        "native_backend_negative_preserved":str(base/"i0/RESULT.json"),"unimplemented_prefix_candidate_count":0,
        "generic":str(base/"g0/RESULT.json"),"complete128":str(base/"s0/RESULT.json"),
        "complete7338":str(base/"c0/RESULT.json"),"actual_interface_recovery_external":str(base/"v0/RESULT.json"),
        "review":str(base/"r0/RESULT.json"),"review_type":reviewer["review_type"],"cost_before_publisher_exit":cost,
        "actual_full_guard_peak_tree_RSS_medians_bytes":reviewer["peak_tree_RSS_medians_bytes"],
        "source_support":"verified_current_Windows_x64_AVX2_OpenCV4point13_IPP_enabled_uint8_contiguous_half_resolution_gray",
        "one_layout_and_native_selection_backend_not_individually_ablated_or_counted_twice":True,
        "R147_R141_related_composition_not_a_second_item":True,"production_default_changed":False,
        "not_verified":reviewer["unverified"]}
    atomic_json(base/"ACCEPTANCE.json",receipt)
    atomic_json(base/"PUBLICATION_001.json",{"status":"scoped_acceptance_published_parent_text_and_final_cost_update_pending",
        "source":identity(Path(__file__)),"reviewer_exact_instances_exited":True,
        "cost_before_publisher_exit":cost,"node_seconds":time.perf_counter()-started,"counted":False})
    print({"status":receipt["status"],"new_optimization_count":1,"complete_phase_reduction":cost["net_reductions"]["phase_seconds"],
        "complete_whole_reduction":cost["net_reductions"]["whole_seconds"]},flush=True)


if __name__=="__main__":
    main()

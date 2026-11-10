"""Audit actual fusion outputs, recorded source revisions, closure, and costs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).parent
OUT=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"
sys.path.insert(0,str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json,data_identity
from run_sift_process_campaign_005 import compare_pair
import fusion_owned_json_write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_matches(path,identity):
    body=Path(path).read_bytes()
    return len(body)==identity["bytes"] and hashlib.sha256(body).hexdigest()==identity["sha256"]


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time()!=row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    own=data_identity(Path(__file__))
    shutil.copyfile(Path(__file__),args.output/"REVIEW_SOURCE.py")
    campaign=read(OUT/"c1/RESULT.json")
    generic=read(OUT/"g1/RESULT.json")
    current={}
    for value in [campaign,generic,*[read(OUT/f"c1/t{i}/{tag}/CONTRACT.json") for i in range(3) for tag in ["b","c"]],
            read(OUT/"r2/error_DRIVER_CONTRACT.json"),read(OUT/"r2/interrupt_DRIVER_CONTRACT.json")]:
        for identity in value.get("sources",[])+value.get("inputs",[])+([value["source"]] if "source" in value else []):
            if identity["path"] in current and identity!=current[identity["path"]]:
                raise AssertionError("A current source or input revision disagrees")
            current[identity["path"]]=identity
    current_checks=[data_identity(Path(p))==identity for p,identity in current.items()]
    copies=[]
    for manifest in [OUT/"g0/RESULT.json",OUT/"g1/RESULT.json",OUT/"c0/PROTOCOL.json",OUT/"c1/RESULT.json",
            *OUT.glob("c*/t*/[bc]/CONTRACT.json"),*OUT.glob("r*/[ei]*/CONTRACT.json"),
            OUT/"r1/recovered/CONTRACT.json",OUT/"r2/recovered/CONTRACT.json"]:
        value=read(manifest)
        for ordinal,identity in enumerate(value.get("sources",[])):
            path=manifest.parent/f"SOURCE_{ordinal:02d}_{Path(identity['path']).name}"
            copies.append({"path":str(path),"recorded_revision_exact":copy_matches(path,identity)})
    for tag in ["error","interrupt"]:
        value=read(OUT/f"r2/{tag}_DRIVER_CONTRACT.json")
        path=OUT/f"r2/{tag}_DRIVER_SOURCE.py"
        copies.append({"path":str(path),"recorded_revision_exact":copy_matches(path,value["source"])})
    for private_source in [OUT/"g1/COMPOSED_OWNED_WRITE_SOURCE.py",*OUT.glob("c1/t*/[bc]/COMPOSED_OWNED_WRITE_SOURCE.py")]:
        if private_source.read_text(encoding="utf-8")!=fusion_owned_json_write.COMPOSED_SOURCE:
            raise AssertionError("A private writer composition changed")
    science=[]
    for i in range(3):
        science.append(compare_pair(OUT/f"c1/t{i}/b",OUT/f"c1/t{i}/c"))
    recovery=compare_pair(OUT/"c1/t0/b",OUT/"r2/recovered")
    windows,linux={},{}
    nodes=[]
    for path in OUT.rglob("NODE.json"):
        value=read(path)
        nodes.append(value)
        for identity in value.get("guard",{}).get("owned_process_identities",[])+([value["owner"]] if "owner" in value else []):
            windows[(identity["pid"],identity["creation_time"])]=identity
        for identity in value.get("Linux_resource_receipts",[]):
            linux[(identity["pid"],identity["creation_time"])]=dict(pid=identity["pid"],creation_time=identity["creation_time"])
        for identity in value.get("actual_Linux_after_wait",[]):
            linux[(identity["pid"],identity["creation_time"])]=dict(pid=identity["pid"],creation_time=identity["creation_time"])
    leases=[]
    for path in sorted(OUT.glob("*_LEASE_*.json")):
        if path.name.startswith("REVIEW_"):
            continue
        value=read(path)
        leases.append({"path":str(path),"released":"lease_released_utc" in value,
            "child_seconds":value["child_elapsed_seconds"],"returncode":value["returncode"]})
        for prefix in ["guard","child"]:
            if prefix+"_pid" in value and prefix+"_creation_time" in value:
                identity=dict(pid=value[prefix+"_pid"],creation_time=value[prefix+"_creation_time"])
                windows[(identity["pid"],identity["creation_time"])]=identity
    for path in OUT.glob("*_GUARD_*.json"):
        for identity in read(path).get("owned_process_identities",[]):
            windows[(identity["pid"],identity["creation_time"])]=identity
    linux_code="import json,psutil; rows=json.loads("+repr(json.dumps(list(linux.values())))+"); out=[]\n"
    linux_code+="for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n out.append(dict(row,alive=alive))\nprint(json.dumps(out))"
    live_linux=json.loads(subprocess.check_output(["wsl","-d","E26-C2-LIV-20260907-134802","--",
        "/opt/e26_thin_surface_v1/venv/bin/python","-B","-c",linux_code],text=True,timeout=30))
    fault_checks=[]
    for tag in ["error","interrupt"]:
        fault=read(OUT/f"r2/{tag}/FAULT_RESULT.json")
        node=read(OUT/f"r2/n{tag}/NODE.json")
        fault_checks.append(fault["status"]=="expected_failure_preserved" and all(fault["checks"].values())
            and node["guard"]["returncode"]==86 and len(fault["traces"]["geometry"]["routes"])==4)
    retry=read(OUT/"RETRY_TOOL_OUTPUT_001.json")
    old_array=read(OUT.parent/"am001/ACCEPTANCE_V2.json")
    old_process=read(OUT.parent/"sp001/ACCEPTANCE.json")
    prior=old_array["complete_first_fees_including_prior_campaign_wrapper_and_V2_review_seconds"]+old_process[
        "complete_first_fees_with_external_and_final_review"]
    side_sum=sum(sum(p["raw_complete_node_and_whole_seconds"].values()) for p in campaign["pairs"])
    all_lease_seconds=sum(row["child_seconds"] for row in leases)
    fees=prior+all_lease_seconds-side_sum
    medians={m:statistics.median(p["raw_complete_node_and_whole_seconds"][m] for p in campaign["pairs"])+fees/6
        for m in ["incumbent","process"]}
    checks={"all_current_source_and_input_identities":all(current_checks),
        "all_saved_source_copies_exact_recorded_revisions":all(r["recorded_revision_exact"] for r in copies),
        "current_joint_generic_and_eight_original_write_guards":generic["status"]=="passed" and all(generic["checks"].values()),
        "three_actual_complete_pairs_full_science_rechecked":len(science)==3 and all(x["all_protected"] for x in science),
        "both_source_bound_actual_two_query_failures_and_hooks":all(fault_checks),
        "existing_output_FileExistsError_preserved":retry["exception_type"]=="FileExistsError"
            and read(OUT/"RETRY_LEASE_001.json")["returncode"]==1 and not (OUT/"r1/nretry/NODE.json").exists(),
        "fresh_complete24_48_recovery_original_outputs_exact":recovery["all_protected"],
        "all_recorded_prior_Windows_instances_gone":all(gone(r) for r in windows.values()),
        "all_recorded_prior_Linux_instances_gone":all(not r["alive"] for r in live_linux),
        "all_previous_leases_released":all(r["released"] for r in leases),
        "complete_node_after_all_prior_fees_at_least5percent":1-medians["process"]/medians["incumbent"]>=.05,
        "each_original_whole_at_most1_1":all(p["raw_complete_node_and_whole_seconds"]["process"]<=
            1.1*p["raw_complete_node_and_whole_seconds"]["incumbent"] for p in campaign["pairs"]),
        "no_new_count_existing_two_mechanisms_only":campaign["new_optimization_count"]==0}
    result={"status":"all_checks_passed_exited_review_fee_pending" if all(checks.values()) else "failed_verification",
        "checks":checks,"current_source_input_identities":list(current.values()),"saved_source_copies":copies,
        "Windows":list(windows.values()),"Linux":live_linux,"prior_leases":leases,
        "original_two_mechanism_first_fees_seconds":prior,"all_prior_nonoverlapping_shared_fees_seconds":fees,
        "raw_complete_side_sum_seconds":side_sum,"complete_medians_with_prior_fees_seconds":medians,
        "whole_and_stage_reduction_with_prior_fees":1-medians["process"]/medians["incumbent"],
        "recovery_invariants":recovery,"root_single_chat_self_review":True,"no_subagent":True,
        "review_before_publication_seconds":time.perf_counter()-started,"source":own,"new_optimization_count":0}
    atomic_json(args.output/"REVIEW.json",result)
    if not all(checks.values()):
        raise AssertionError("A fusion source, recovery, cost or closure gate failed")


if __name__=="__main__":
    main()

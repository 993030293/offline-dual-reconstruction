"""Measure complete protected localization with prepared geometry common."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"
sys.path.insert(0,str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_sift_process_campaign_005 import compare_pair


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    generic = read(OWNED/"g1/RESULT.json")
    if generic["status"] != "passed" or not all(generic["checks"].values()):
        raise ValueError("The coupled component contract has not passed")
    paths = [Path(__file__),CODE/"geometry_sift_fusion_node.py",CODE/"localize_geometry_sift_fusion.py",
        CODE/"check_geometry_sift_fusion.py",CODE/"native_sift_process.py",CODE/"array_match_geometry.py",
        CODE/"sift_process_node_003.py",CODE/"run_sift_process_campaign_005.py",CODE/"run_bounded_journal_task.py",
        CODE/"owned_json_write.py",CODE/"fusion_owned_json_write.py",
        ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/compare_actual_localization.py"]
    sources = [data_identity(p) for p in paths]
    for recorded in generic["sources"]:
        if data_identity(Path(recorded["path"])) != recorded:
            raise ValueError("A source changed after the joint component check")
    for i,path in enumerate(paths):
        shutil.copyfile(path,args.output/f"SOURCE_{i:02d}_{path.name}")
    old_array = read(OWNED.parent/"am001/ACCEPTANCE_V2.json")
    old_process = read(OWNED.parent/"sp001/ACCEPTANCE.json")
    if not all(old_array["checks"].values()) or not all(old_process["checks"].values()):
        raise ValueError("Both original scoped mechanism acceptances are required")
    prior = old_array["complete_first_fees_including_prior_campaign_wrapper_and_V2_review_seconds"] + old_process[
        "complete_first_fees_with_external_and_final_review"]
    state = {"status":"running","sources":sources,"orders":[["incumbent","process"],["process","incumbent"],["incumbent","process"]],
        "pairs":[],"runs":[],"new_optimization_count":0,"prior_two_mechanism_first_fees_seconds":prior,
        "prepared_geometry_common":True,"complete_node_and_whole_same_parent_window":True,
        "minimum_complete_reduction":0.05,"maximum_each_whole_ratio":1.1}
    atomic_json(args.output/"PROTOCOL.json",state)
    try:
        for trial,order in enumerate(state["orders"]):
            folder = args.output/f"t{trial}"
            folder.mkdir()
            costs={}
            for mode in order:
                if time.perf_counter()-started>1700 or sources != [data_identity(p) for p in paths]:
                    raise RuntimeError("A frozen campaign source or duration changed")
                tag = "b" if mode=="incumbent" else "c"
                node,output = folder/("n"+tag),folder/tag
                command = [sys.executable,"-B",str(CODE/"geometry_sift_fusion_node.py"),"--kind","localization",
                    "--mode",mode,"--node",str(node),"--output",str(output),"--",sys.executable,"-B",
                    str(CODE/"localize_geometry_sift_fusion.py"),"--mode",mode,"--output",str(output)]
                row={"trial":trial,"mode":mode,"command":command,"status":"running"}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json",state)
                before = time.perf_counter()
                with (folder/(mode+"_controller.log")).open("x",encoding="utf-8") as log:
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    row["pid"]=process.pid
                    atomic_json(args.output/"STATE.json",state)
                    code = process.wait(timeout=1800)
                receipt=read(node/"NODE.json")
                result=read(output/"RESULT.json") if (output/"RESULT.json").exists() else None
                row.update(status="completed" if code==0 else "failed",returncode=code)
                atomic_json(args.output/"STATE.json",state)
                if code or receipt["status"]!="completed" or receipt["live_Windows_after_wait"] or any(
                        x["alive"] for x in receipt["actual_Linux_after_wait"]) or not result or not all(result["checks"].values()):
                    raise RuntimeError("An original full localization node failed")
                costs[mode]=time.perf_counter()-before
                row["complete_parent_window_seconds"]=costs[mode]
                atomic_json(args.output/"STATE.json",state)
            protected=compare_pair(folder/"b",folder/"c")
            atomic_json(folder/"INVARIANTS.json",protected)
            checks={"all_original_complete_scientific_outputs_exact":protected["all_protected"],
                "each_whole_at_most1_1":costs["process"]<=1.1*costs["incumbent"]}
            pair={"trial":trial,"order":order,"checks":checks,"raw_complete_node_and_whole_seconds":costs,
                "complete_compare_seconds":protected["complete_compare_seconds"]}
            state["pairs"].append(pair)
            atomic_json(args.output/"STATE.json",state)
            print(json.dumps(pair),flush=True)
            if not all(checks.values()):
                raise ValueError("A complete science or whole-node guard failed")
        state.update(status="three_complete_pairs_pending_all_fees_recovery_and_self_review",
            raw_medians_seconds={m:statistics.median(p["raw_complete_node_and_whole_seconds"][m] for p in state["pairs"])
                for m in ["incumbent","process"]},complete_campaign_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json",state)
    except BaseException as error:
        state.update(status="failed",error=repr(error),complete_campaign_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json",state)
        raise
    finally:
        atomic_json(args.output/"STATE.json",state)


if __name__ == "__main__":
    main()

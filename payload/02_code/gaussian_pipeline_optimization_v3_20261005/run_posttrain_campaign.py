"""Compare complete serial and parallel posttraining DAGs and owned restart behavior."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def metrics(path):
    result = read(path/"pair/evaluation/EVALUATION.json")
    return {label: row["rows"] for label, row in result["results"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair-source", "checkpoint", "runtime", "reference-run", "localization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("posttrain_dag.py"), Path(__file__).with_name("evaluate_dag_models.py")]
    identity = [data_identity(p) for p in sources]
    state = {"status": "running", "sources": identity, "pairs": [], "minimum_whole_makespan_reduction": .1,
        "all24_metrics_exact_same_two12000_models": True, "complete_native_archive_bytes_exact": True,
        "prepare_copy_and_verification_in_whole_makespan": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    import shutil
    (args.output/"sources").mkdir()
    for source in sources:
        shutil.copyfile(source, args.output/"sources"/source.name)

    def command(mode, output):
        result = [sys.executable, str(Path(__file__).with_name("posttrain_dag.py")), "--mode", mode, "--output", str(output)]
        for name in ("pair_source", "checkpoint", "runtime", "reference_run", "localization"):
            result += ["--"+name.replace("_", "-"), str(getattr(args, name))]
        return result

    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["serial", "parallel"] if trial % 2 == 0 else ["parallel", "serial"]):
                subprocess.run(command(mode, pair/mode), check=True)
            completed = {m: read(pair/m/"RESULT.json") for m in ("serial", "parallel")}
            archives = [data_identity(pair/m/"archive/checkpoint.archive") for m in ("serial", "parallel")]
            checks = {"both_actual_DAGs_published_only_after_completion": all(r["publication_created"] for r in completed.values()),
                "all24_two_model_metric_rows_exact": metrics(pair/"serial") == metrics(pair/"parallel"),
                "same_verified_native_archive_bytes": archives[0]["sha256"] == archives[1]["sha256"],
                "actual_sixview_RGB_artifact_exact": data_identity(pair/"serial/pair/evaluation/SIX_VIEW_COMPARISON.jpg")["sha256"]
                    == data_identity(pair/"parallel/pair/evaluation/SIX_VIEW_COMPARISON.jpg")["sha256"],
                "only_one_GPU_task": all(read(pair/m/"PROTOCOL.json")["GPU_heavy_tasks"] == 1 for m in completed)}
            row = {"trial": trial, "checks": checks, "makespan_seconds": {m: r["whole_makespan_seconds"] for m, r in completed.items()},
                "conservative_sum_of_task_peak_RSS_bytes": {m: r["conservative_sum_of_task_peak_RSS_bytes"] for m, r in completed.items()}}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("A complete finalization DAG output or dependency guard failed")
        interrupted = args.output/"lifecycle"
        code = subprocess.run(command("serial", interrupted)+["--interrupt-after-evaluation"]).returncode
        before = read(interrupted/"STATE.json")
        frozen = before["tasks"]["evaluate"]["payload"]
        mtime = (interrupted/"pair/evaluation/EVALUATION.json").stat().st_mtime_ns
        lifecycle = {"injected_owned_stop_returned86": code == 86,
            "incomplete_DAG_not_published": not (interrupted/"PUBLICATION.json").exists(),
            "evaluation_complete_archive_not_started": set(before["tasks"]) == {"evaluate"} and before["tasks"]["evaluate"]["status"] == "complete"}
        subprocess.run(command("serial", interrupted)+["--resume"], check=True)
        after = read(interrupted/"RESULT.json")
        lifecycle.update(exact_completed_evaluation_reused_without_reexecution=after["tasks"]["evaluate"]["payload"] == frozen
            and (interrupted/"pair/evaluation/EVALUATION.json").stat().st_mtime_ns == mtime,
            resume_published_after_both_completed=after["publication_created"] and after["tasks"]["archive"]["status"] == "complete")
        atomic_json(args.output/"LIFECYCLE.json", lifecycle)
        medians = {m: statistics.median(r["makespan_seconds"][m] for r in state["pairs"]) for m in ("serial", "parallel")}
        reduction = 1-medians["parallel"]/medians["serial"]
        checks = {"all_three_pairs_same_complete_tasks_and_all_outputs": len(state["pairs"]) == 3 and all(all(r["checks"].values()) for r in state["pairs"]),
            "actual_makespan_reduction_at_least10percent": reduction >= .1,
            "actual_interruption_and_restart_dependency_guards": all(lifecycle.values()),
            "sources_unchanged": identity == [data_identity(p) for p in sources]}
        state.update(status="all_actual_DAG_guards_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, median_makespan_seconds=medians, reduction=reduction, lifecycle=lifecycle)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_no_acceptance", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()

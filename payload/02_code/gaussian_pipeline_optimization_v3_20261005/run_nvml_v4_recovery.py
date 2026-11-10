"""Exercise an actual owned-node query fault and fresh complete localization recovery."""

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nm_v4_001"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nm001"
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def load_private(filename, name):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fault", "recover", "fault-child", "recover-child"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    if args.mode == "fault-child":
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "WORKER_STARTED.json", {"pid": os.getpid(), "creation_time": psutil.Process().create_time()})
        time.sleep(90)
        raise AssertionError("The deliberate monitor fault did not drain its actual owned child")
    if args.mode == "recover-child":
        module = load_private("localize_monitor_reference.py", "private_v4_complete_localizer")
        module.OWNED = OWN
        previous = sys.argv
        try:
            sys.argv = [str(CODE / "localize_monitor_reference.py"), "--mode", "incumbent", "--output", str(output)]
            module.main()
        finally:
            sys.argv = previous
        return

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "executed_sources"
    snapshot.mkdir()
    sources = [Path(__file__), CODE / "nvml_owned_node.py", CODE / "nvml_memory_monitor_002.py",
               CODE / "localize_monitor_reference.py", CODE / "nvml_node_config.py", V1 / "run_bounded_task.py"]
    identities = [data_identity(path) for path in sources]
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshot / (str(index) + "_" + path.name))
    metadata_folder = OWN / "p1"
    metadata_folder.mkdir(exist_ok=True)
    metadata_path = metadata_folder / "RESULT.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != json.loads((OLD / "p1/RESULT.json").read_text()):
            raise AssertionError("The private original NVML metadata contract changed")
    else:
        shutil.copyfile(OLD / "p1/RESULT.json", metadata_path)
    module = load_private("nvml_owned_node.py", "private_v4_owned_controller")
    module.OUT = OWN
    original_memory = module.NativeMemory
    child_output = output / "task"
    injection = []
    if args.mode == "fault":
        class FaultMemory(original_memory):
            """Propagate one declared API-interface fault after a real native sample."""

            def mib(self, version):
                value = super().mib(version)
                deadline = time.monotonic() + 5
                marker = child_output / "WORKER_STARTED.json"
                while not marker.exists():
                    if time.monotonic() >= deadline:
                        raise AssertionError("The actual controlled child never started")
                    time.sleep(.01)
                injection.append({"actual_real_sample": list(value), "worker": json.loads(marker.read_text()),
                                  "deliberate_interface_error_code": 999, "actual_driver_fault": False})
                self.check(999)
                raise AssertionError("The original NVML status propagation was bypassed")
        module.NativeMemory = FaultMemory
    # A child can be born inside the first sampling call after the initial census.
    original_main = module.main
    source = inspect.getsource(original_main)
    anchor = '            record["controller_error"] = repr(error)\n            victims = []'
    replacement = '''            record["controller_error"] = repr(error)
            guard_path = args.node / "guard.json"
            if guard_path.is_file():
                for item in safe_json(guard_path).get("owned_process_identities", []):
                    tracked[item["pid"]] = item["creation_time"]
            try:
                parent = psutil.Process(owner["pid"])
                if parent.create_time() != owner["creation_time"]:
                    raise RuntimeError("The owned guard identity changed during fault drain")
                for item in parent.children(recursive=True):
                    tracked[item.pid] = item.create_time()
            except psutil.NoSuchProcess:
                pass
            victims = []'''
    if source.count(anchor) != 1:
        raise AssertionError("The exact original owned-controller fault source changed")
    prepared_main = source.replace(anchor, replacement)
    executed_main = output / "PRIVATE_CONTROLLER_EXECUTED_MAIN.py"
    executed_main.write_text(prepared_main, encoding="utf-8")
    namespace = dict(original_main.__globals__)
    exec(compile(prepared_main, str(executed_main), "exec"), namespace)
    module.main = namespace[original_main.__name__]
    atomic_json(output / "PRIVATE_CONTROLLER_BINDING.json", {
        "original_source": data_identity(CODE / "nvml_owned_node.py"),
        "actual_executed_main": data_identity(executed_main),
        "only_exception_path_owned_census_corrected": True,
        "common_for_cli_and_nvml": True, "new_optimization_count": 0})
    previous = sys.argv
    expected_exit = None
    try:
        sys.argv = [str(CODE / "nvml_owned_node.py"), "--kind", "localization", "--mode", "nvml",
                    "--node", str(output / "node"), "--output", str(child_output), "--",
                    sys.executable, "-X", "utf8", str(Path(__file__)),
                    "--mode", args.mode + "-child", "--output", str(child_output)]
        try:
            module.main()
        except SystemExit as error:
            if args.mode != "fault" or error.code != 1:
                raise
            expected_exit = 1
    finally:
        sys.argv = previous
        module.NativeMemory = original_memory
        module.main = original_main
        namespace["NativeMemory"] = original_memory
    node = json.loads((output / "node/NODE.json").read_text())
    checks = {"all_original_node_connections_and_owned_processes_closed": not node["live_Windows_after_wait"]
              and all(not row["alive"] for row in node["actual_Linux_after_wait"]),
              "original_NVML_session_shutdown_and_source_unchanged": node["monitoring"]["library_closed"]
              and node["monitoring"]["source_unchanged"],
              "all_wrapper_sources_unchanged": identities == [data_identity(path) for path in sources],
              "private_alias_and_argv_restored": module.NativeMemory is original_memory and sys.argv is previous,
              "no_production_source_or_input_modified": node["final_source_unchanged"]}
    if args.mode == "fault":
        checks["actual_started_child_and_real_NVML_sample_before_interface_fault"] = len(injection) == 1
        checks["original_status_error_propagated_no_silent_fallback"] = expected_exit == 1 and node["status"] == "failed" and "status 999" in node["controller_error"]
        checks["actual_controlled_worker_PID_creation_identity_gone"] = not module.remaining([injection[0]["worker"]])
    else:
        from run_nvml_monitor_campaign_002 import compare_pair
        compared = compare_pair(OLD / "c1/t0/b", child_output)
        atomic_json(output / "INVARIANTS.json", compared)
        checks["actual_fresh_complete24_48_recovered_to_original_scientific_outputs"] = node["status"] == "completed" and compared["all_protected"]
        checks["prior_interface_fault_preserved"] = json.loads((OWN / "f1/RESULT.json").read_text())["status"] == "actual_original_monitor_interface_fault_propagated_and_owned_child_closed"
    if not all(checks.values()):
        raise AssertionError("An actual original monitor fault or recovery guard did not pass")
    result = {"status": "actual_original_monitor_interface_fault_propagated_and_owned_child_closed" if args.mode == "fault"
              else "actual_fresh_original24_48_recovery_passed", "mode": args.mode, "checks": checks,
              "sources": identities, "injection": injection, "node": str(output / "node/NODE.json"),
              "new_optimization_count": 0, "complete_seconds": time.perf_counter() - started,
              "fault_classification": "deliberate_interface_fixture_after_real_driver_query_not_actual_driver_failure"}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "seconds": result["complete_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

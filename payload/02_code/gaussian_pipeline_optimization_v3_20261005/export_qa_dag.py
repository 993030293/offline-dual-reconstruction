"""Overlap immutable-map verification with owned original artifact operations."""

from contextlib import contextmanager
import builtins
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from segmented_export_execution import BEFORE, AFTER, FINAL
from runtime import atomic_json, data_identity

FILTER_BOUNDARY = '        removed_after = filter_points(model, pp, lo, hi)'
PERSIST = '''        filtered = args.output/"training_opencv_model"
        filtered.mkdir()
        model.write(filtered)'''
UNDISTORT = '''        pycolmap.undistort_images(dataset, filtered, args.images, image_names=sorted(keep),
            output_type="COLMAP", num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)'''
ARTIFACTS = '''        sparse = dataset/"sparse/0"
        sparse.mkdir()
        pinhole.write(sparse)
        pinhole.write_text(sparse)
        write_initial_ply(pinhole, sparse/"points3D.ply")'''


class OwnedArtifacts:
    """Keep one native process and defer preparation errors to the original boundary."""

    def __init__(self, model, filtered, dataset, images, names, output, native):
        self.output, self.process, self.start_error = Path(output), None, None
        self.trace = {"status": "preparing", "native_original_operations": True, "prepare_seconds": None}
        started = time.perf_counter()
        try:
            self.output.mkdir(parents=True, exist_ok=False)
            filtered.mkdir()
            model.write(filtered)
            code = Path(__file__).resolve().parent
            sources = [code / "export_qa_artifact_worker.py", Path(native.__file__), Path(inspect.getfile(native.projection_validity)),
                       Path(inspect.getfile(data_identity))]
            contract = {"dataset": str(dataset), "filtered": str(filtered), "images": str(images), "image_names": list(names),
                        "sources": [data_identity(path) for path in dict.fromkeys(sources)],
                        "map_inputs": [data_identity(path) for path in sorted(filtered.glob("*.bin"))], "new_optimization_count": 0}
            atomic_json(self.output / "CONTRACT.json", contract)
            self.log = (self.output / "WORKER.log").open("x", encoding="utf-8")
            command = [sys.executable, "-B", str(code / "export_qa_artifact_worker.py"), "--contract", str(self.output / "CONTRACT.json")]
            atomic_json(self.output / "COMMAND.json", {"command": command, "new_optimization_count": 0})
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
            self.identity = {"pid": self.process.pid, "creation_time": psutil.Process(self.process.pid).create_time()}
            self.trace.update(status="running", identity=self.identity)
        except Exception as error:
            self.start_error = error
            self.trace.update(status="prepare_failed_deferred_to_original_boundary", error=repr(error))
        finally:
            self.trace["prepare_seconds"] = time.perf_counter() - started

    def commit_preparation(self):
        if self.start_error is not None:
            raise self.start_error

    def raise_worker_error(self):
        path = self.output / "FAILURE.json"
        if path.exists():
            failure = json.loads(path.read_text(encoding="utf-8"))
            allowed = {name: getattr(builtins, name) for name in ("ValueError", "RuntimeError", "MemoryError", "OSError", "FileNotFoundError", "FileExistsError", "PermissionError")}
            exception = allowed.get(failure["error_type"], RuntimeError)
            raise exception(failure["error"])
        raise RuntimeError("The owned native artifact process exited without its required result")

    def wait_undistorted(self):
        self.commit_preparation()
        before = time.perf_counter()
        while not (self.output / "UNDISTORT_READY.json").exists():
            if self.process.poll() is not None:
                self.raise_worker_error()
            time.sleep(.01)
        ready = json.loads((self.output / "UNDISTORT_READY.json").read_text(encoding="utf-8"))
        if ready["status"] != "original_undistort_returned":
            raise ValueError("A ready marker cannot admit an incomplete original undistortion")
        self.trace.update(undistort_wait_seconds=time.perf_counter() - before, undistort_ready=ready)

    def wait_artifacts(self):
        self.commit_preparation()
        before = time.perf_counter()
        returncode = self.process.wait()
        if returncode != 0 or not (self.output / "RESULT.json").exists():
            self.raise_worker_error()
        result = json.loads((self.output / "RESULT.json").read_text(encoding="utf-8"))
        if result["status"] != "complete_original_undistort_and_sparse_artifacts" or not result["source_and_private_map_unchanged"]:
            raise ValueError("The complete original worker output protection failed")
        self.trace.update(status="completed_and_joined", artifact_wait_and_join_seconds=time.perf_counter() - before,
                          worker_result=result, returncode=returncode)

    def close(self, exceptional):
        before = time.perf_counter()
        if self.process is not None:
            if self.process.poll() is None:
                if not exceptional:
                    raise RuntimeError("A normal export cannot leave its owned worker running")
                if psutil.Process(self.process.pid).create_time() != self.identity["creation_time"]:
                    raise RuntimeError("An owned worker PID identity changed before cancellation")
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
                self.trace["cancelled_and_joined"] = True
            self.trace["closed_returncode"] = self.process.returncode
        if hasattr(self, "log"):
            self.log.close()
        self.trace.update(closed=True, close_seconds=time.perf_counter() - before)
        if self.output.exists():
            atomic_json(self.output / "CLOSURE.json", self.trace)


def composed_source(original_source, mode):
    anchors = (BEFORE, AFTER, FINAL, FILTER_BOUNDARY, PERSIST, UNDISTORT, ARTIFACTS)
    if any(original_source.count(anchor) != 1 for anchor in anchors):
        raise ValueError("The original immutable-state export boundaries changed")
    if mode not in ("serial", "dag"):
        raise ValueError("An explicit serial or DAG execution is required")
    source = original_source.replace(BEFORE,
        '        support_before = _r132_support(model, policy["camera_geometry_admission"], camera_geometry_metrics, "before", True)')
    source = source.replace(AFTER,
        '        support_after = _r132_support(model, policy["camera_geometry_admission"], camera_geometry_metrics, "after", True)')
    source = source.replace(FINAL,
        '        final_support = _r132_support(pinhole, policy["camera_geometry_admission"], camera_geometry_metrics, "final", False)')
    if mode == "serial":
        source = source.replace(FILTER_BOUNDARY, FILTER_BOUNDARY + '\n        _r140_region_start()')
        source = source.replace(ARTIFACTS, ARTIFACTS + '\n        _r140_region_end()')
    else:
        source = source.replace(PERSIST, '        _r140_commit(worker)')
        source = source.replace(FILTER_BOUNDARY, FILTER_BOUNDARY + '''
        _r140_region_start()
        filtered = args.output/"training_opencv_model"
        worker = _r140_prepare(model, filtered, args.output/"dataset", args.images, sorted(keep))''')
        source = source.replace(UNDISTORT, '        _r140_wait_undistorted(worker)')
        source = source.replace(ARTIFACTS, '''        sparse = dataset/"sparse/0"
        _r140_wait_artifacts(worker)
        _r140_region_end()''')
    return source


@contextmanager
def export_qa_dag_execution(native, original_source, mode, output):
    before = time.perf_counter()
    original = native.main
    namespace = dict(original.__globals__)
    if "_r132_support" not in namespace:
        raise ValueError("The common accepted whole-map support execution must be installed first")
    trace = {"mode": mode, "setup_seconds": None, "restore_seconds": None, "region_start_count": 0,
             "region_end_count": 0, "worker": None, "original_binding_restored": False}
    worker = None
    region_started = None

    def start():
        nonlocal region_started
        trace["region_start_count"] += 1
        if region_started is not None:
            raise ValueError("The full affected region cannot start twice")
        region_started = time.perf_counter()

    def end():
        if region_started is None or trace["region_end_count"] != 0:
            raise ValueError("The full affected region requires one start and one end")
        trace["affected_region_seconds"] = time.perf_counter() - region_started
        trace["region_end_count"] += 1

    def prepare(model, filtered, dataset, images, names):
        nonlocal worker
        if worker is not None:
            raise ValueError("Only one owned native artifact worker is admitted")
        worker = OwnedArtifacts(model, filtered, dataset, images, names, output, native)
        return worker

    namespace.update(_r140_region_start=start, _r140_region_end=end, _r140_prepare=prepare,
        _r140_commit=lambda worker: worker.commit_preparation(),
        _r140_wait_undistorted=lambda worker: worker.wait_undistorted(),
        _r140_wait_artifacts=lambda worker: worker.wait_artifacts())
    source = composed_source(original_source, mode)
    exec(compile(source, "<private_original_export_QA_artifact_DAG>", "exec"), namespace)
    exceptional = True
    try:
        native.main = namespace[original.__name__]
        trace["setup_seconds"] = time.perf_counter() - before
        yield trace
        exceptional = False
    finally:
        restoring = time.perf_counter()
        try:
            if worker is not None:
                worker.close(exceptional)
                trace["worker"] = worker.trace
        finally:
            native.main = original
            trace.update(restore_seconds=time.perf_counter() - restoring, original_binding_restored=native.main is original)

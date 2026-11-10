"""Check non-box maps, original worker failures, cancellation and full fresh recovery."""

import time
STARTED = time.perf_counter()

import argparse
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import psutil
from PIL import Image

from profile_export_qa_dag import ROOT, BASE, V1, V1OUT, native
from profile_grid_support import original_metric_module
from profile_sparse_ply_oracle import compare_outputs
from grid_support import grid_support_execution
from segmented_export_execution import segmented_export_execution
from export_qa_dag import OwnedArtifacts, export_qa_dag_execution, composed_source
from machine_json import exact_json_values
from runtime import atomic_json, data_identity

CODE = Path(__file__).resolve().parent
FIXTURES = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/generic_binary_qa_001"
PUBLIC = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def gone(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def component(folder, model_source, photos, mode, module, policy, expected_formats=("PINHOLE",)):
    folder.mkdir(parents=True, exist_ok=False)
    export = folder / "export"
    export.mkdir()
    model = native.pycolmap.Reconstruction(model_source)
    names = sorted(image.name for image in model.images.values() if image.has_pose)
    filtered, dataset = export / "training_opencv_model", export / "dataset"
    worker = None
    before = time.perf_counter()
    try:
        if mode == "dag":
            worker = OwnedArtifacts(model, filtered, dataset, photos, names, folder / "worker", native)
        initial = native.projection_validity(model)
        support_initial = [module.camera_geometry_metrics(model, image, policy) for image in model.images.values() if image.has_pose]
        if not initial["valid"]:
            raise ValueError("A fixed component map has invalid original observed projections")
        if mode == "serial":
            filtered.mkdir()
            model.write(filtered)
            native.pycolmap.undistort_images(dataset, filtered, photos, image_names=names,
                output_type="COLMAP", num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        else:
            worker.commit_preparation()
            worker.wait_undistorted()
        pinhole = native.pycolmap.Reconstruction(dataset / "sparse")
        photo_names = sorted(path.name for path in (dataset / "images").iterdir() if path.is_file())
        if photo_names != names or sorted(image.name for image in pinhole.images.values() if image.has_pose) != names:
            raise ValueError("The complete component image or camera membership changed")
        if any(camera.model_name not in expected_formats for camera in pinhole.cameras.values()) or set(model.points3D) != set(pinhole.points3D):
            raise ValueError("The component camera format or full point membership changed")
        pose_delta, point_delta = 0., 0.
        for image in pinhole.images.values():
            if image.has_pose:
                pose_delta = max(pose_delta, float(np.max(np.abs(image.cam_from_world().matrix() - model.images[image.image_id].cam_from_world().matrix()))))
                with Image.open(dataset / "images" / image.name) as photo:
                    camera = pinhole.cameras[image.camera_id]
                    if photo.size != (camera.width, camera.height):
                        raise ValueError("The complete component photo dimensions changed")
        for point_id, point in pinhole.points3D.items():
            point_delta = max(point_delta, float(np.linalg.norm(point.xyz - model.points3D[point_id].xyz)))
        final = native.projection_validity(pinhole)
        support_final = [module.camera_geometry_metrics(pinhole, image, policy) for image in pinhole.images.values() if image.has_pose]
        if max(pose_delta, point_delta) > 1e-9 or not final["valid"]:
            raise ValueError("The complete original component projection or metric coordinates changed")
        if mode == "serial":
            sparse = dataset / "sparse/0"
            sparse.mkdir()
            pinhole.write(sparse)
            pinhole.write_text(sparse)
            native.write_initial_ply(pinhole, sparse / "points3D.ply")
        else:
            worker.wait_artifacts()
        seconds = time.perf_counter() - before
        result = {"initial_projection": initial, "final_projection": final, "support_initial": support_initial,
            "support_final": support_final, "registered": pinhole.num_reg_images(), "points": pinhole.num_points3D(),
            "observations": pinhole.compute_num_observations(), "pose_delta": pose_delta, "point_delta": point_delta,
            "native_camera_models": sorted(camera.model_name for camera in pinhole.cameras.values())}
        return result, seconds
    finally:
        if worker is not None:
            worker.close(sys.exc_info()[0] is not None)
            if not gone(worker.identity):
                raise RuntimeError("The private component worker remained alive after close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BASE)
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    module, common = original_metric_module()
    policy_path = common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"
    policy = read(policy_path)["camera_geometry_admission"]
    fixtures = sorted(path for path in FIXTURES.iterdir() if path.is_dir() and (path / "images.bin").is_file())
    if len(fixtures) != 9:
        raise ValueError("All nine fixed non-box geometry and scale fixtures are required")
    photos = output / "synthetic_photos"
    photos.mkdir()
    first = native.pycolmap.Reconstruction(fixtures[0])
    for image in first.images.values():
        camera = first.cameras[image.camera_id]
        y, x = np.indices((camera.height, camera.width), dtype=np.uint32)
        pixels = np.stack(((x + 2*y) % 256, (3*x + y + 17) % 256, (x + y + 31) % 256), axis=2).astype(np.uint8)
        Image.fromarray(pixels).save(photos / image.name)
    paths = [Path(__file__), CODE / "export_qa_dag.py", CODE / "export_qa_artifact_worker.py", CODE / "profile_export_qa_dag.py",
             CODE / "segmented_export_execution.py", CODE / "segmented_camera_support.py", Path(native.__file__), V1 / "run_metric_refinement.py", policy_path]
    sources = [data_identity(path) for path in paths]
    inputs = [data_identity(path) for folder in fixtures for path in sorted(folder.glob("*.bin"))]
    inputs += [data_identity(path) for path in sorted((PUBLIC / "sparse/0").glob("*.bin"))]
    public_photos = sorted((PUBLIC / "images").glob("*.png"))
    if len(public_photos) != 360:
        raise ValueError("The full existing public 360-image inventory changed")
    inputs += [data_identity(path) for path in public_photos]
    copies = output / "source_snapshot"
    copies.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": sources, "inputs": inputs,
        "all_nine_fixed_geometries_scales_three_camera_models": True, "all360_public_photos_component_only": True,
        "three_real_full_entry_exceptions_and_complete_new_directory_recovery": True, "new_optimization_count": 0})
    checks, records = {}, []
    original_main, original_projection, original_argv = native.main, native.projection_validity, sys.argv
    original_source, metric = inspect.getsource(native.main), module.camera_geometry_metrics
    for mode in ("serial", "dag"):
        changed = original_source.replace("jpeg_quality=97", "jpeg_quality=96")
        caught = None
        try:
            composed_source(changed, mode)
        except ValueError as error:
            caught = error
        checks[f"changed_original_source_boundary_rejected_{mode}"] = caught is not None
    try:
        for index, fixture in enumerate(fixtures):
            pair = output / f"generic{index}"
            values = {}
            seconds = {}
            for mode in ("serial", "dag"):
                values[mode], seconds[mode] = component(pair / mode, fixture, photos, mode, module, policy)
            exact_json_values(values["serial"], values["dag"])
            comparison = compare_outputs(pair / "serial/export", pair / "dag/export")
            checks[f"full_geometry_scale_calibration_artifacts_exact_{index}"] = values["dag"]["registered"] == 9
            checks[f"full1440_observations160_points_original_QA_exact_{index}"] = values["dag"]["observations"] == 1440 and values["dag"]["points"] == 160
            records.append({"fixture": fixture.name, "comparison": comparison, "complete_component_seconds": seconds})
        pair = output / "public360"
        values, seconds = {}, {}
        public_source = native.pycolmap.Reconstruction(PUBLIC / "sparse/0")
        if any(camera.model_name != "OPENCV" or len(camera.params) != 8 or np.any(camera.params[4:] != 0)
               for camera in public_source.cameras.values()):
            raise ValueError("The diagnosed original zero-distortion public camera contract changed")
        for mode in ("serial", "dag"):
            values[mode], seconds[mode] = component(pair / mode, PUBLIC / "sparse/0", PUBLIC / "images", mode, module, policy, ("OPENCV",))
        exact_json_values(values["serial"], values["dag"])
        comparison = compare_outputs(pair / "serial/export", pair / "dag/export")
        checks["all360_public_members133020_points848219_observations_exact"] = (values["dag"]["registered"], values["dag"]["points"], values["dag"]["observations"]) == (360, 133020, 848219)
        records.append({"fixture": "existing_Oxford360_component_development_not_blind", "comparison": comparison, "complete_component_seconds": seconds,
            "native_camera_models": values["dag"]["native_camera_models"], "strict_current_PINHOLE_training_admission": "not_passed_original_native_OPENCV_remains"})
        ba, roles, images = V1OUT / "new_map_metric_BA_001", V1OUT / "new_map_roles_postBA_001/TRAINING_ROLES.json", V1OUT / "e2_geometry_support_001/images"
        for exception in (RuntimeError, KeyboardInterrupt, SystemExit):
            target = output / ("fault_" + exception.__name__)
            target.mkdir()
            def interrupted(model, exception=exception):
                raise exception("injected original full projection failure")
            caught, trace = None, None
            native.projection_validity = interrupted
            try:
                with grid_support_execution(module, True):
                    with segmented_export_execution(native, target / "support_states", True):
                        with export_qa_dag_execution(native, original_source, "dag", target / "worker") as trace:
                            sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles), "--images", str(images), "--output", str(target / "export")]
                            native.main()
            except BaseException as error:
                caught = error
            finally:
                sys.argv, native.projection_validity = original_argv, original_projection
            state, admission = read(target / "export/STATE.json"), read(target / "export/TRAINING_ADMISSION.json")
            checks[exception.__name__ + "_exact_original_error_propagated"] = type(caught) is exception and str(caught) == "injected original full projection failure"
            checks[exception.__name__ + "_actual_worker_cancelled_joined_no_admission"] = trace["worker"]["cancelled_and_joined"] and gone(trace["worker"]["identity"])
            checks[exception.__name__ + "_failed_state_and_original_bindings_restored"] = state["status"] == "failed_export_no_training_admission"
            checks[exception.__name__ + "_failed_state_and_original_bindings_restored"] &= admission["status"] == "subset_checks_pending" and native.main is original_main and module.camera_geometry_metrics is metric
            records.append({"injected_exception": exception.__name__, "trace": trace})
        blocked = output / "existing_filtered_directory"
        (blocked / "export/training_opencv_model").mkdir(parents=True)
        model = native.pycolmap.Reconstruction(fixtures[0])
        worker = OwnedArtifacts(model, blocked / "export/training_opencv_model", blocked / "export/dataset", photos,
            sorted(image.name for image in model.images.values() if image.has_pose), blocked / "worker", native)
        caught = None
        try:
            worker.commit_preparation()
        except FileExistsError as error:
            caught = error
        finally:
            worker.close(True)
        checks["actual_filtered_directory_error_deferred_and_propagated_no_worker"] = caught is not None and worker.process is None and worker.trace["closed"]
        blocked = output / "existing_artifact_directory"
        (blocked / "export/dataset/sparse/0").mkdir(parents=True)
        worker = OwnedArtifacts(model, blocked / "export/training_opencv_model", blocked / "export/dataset", photos,
            sorted(image.name for image in model.images.values() if image.has_pose), blocked / "worker", native)
        caught = None
        try:
            worker.commit_preparation()
            worker.wait_undistorted()
            worker.wait_artifacts()
        except FileExistsError as error:
            caught = error
        finally:
            worker.close(True)
        checks["actual_artifact_directory_process_error_propagated_and_joined"] = caught is not None and gone(worker.identity)
        recovery = output / "complete_fresh_recovery"
        command = [sys.executable, "-B", str(CODE / "profile_export_qa_dag.py"), "--mode", "dag", "--output", str(recovery)]
        atomic_json(output / "RECOVERY_COMMAND.json", {"command": command, "new_optimization_count": 0})
        before = time.perf_counter()
        with (output / "RECOVERY.log").open("x", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        recovery_seconds = time.perf_counter() - before
        recovered = read(recovery / "RESULT.json")
        comparison = compare_outputs(BASE / "c0/t0/serial/export", recovery / "export")
        checks["actual_fresh_full1016_recovery_all_original1050_outputs_exact"] = all(recovered["checks"].values()) and comparison["complete_byte_pairs"] == 1046 and comparison["complete_JSON_pairs"] == 4
        records.append({"complete_new_directory_recovery_seconds": recovery_seconds, "comparison": comparison})
        checks["all_current_sources_and_all_generic_public_inputs_unchanged"] = sources == [data_identity(row["path"]) for row in sources] and inputs == [data_identity(row["path"]) for row in inputs]
        checks["all_original_private_bindings_restored"] = native.main is original_main and native.projection_validity is original_projection and sys.argv is original_argv and module.camera_geometry_metrics is metric
        checks["all_owned_disk_bounds"] = sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) < 30 * 1024**3 and shutil.disk_usage(ROOT).free >= 20 * 1024**3
        result = {"status": "complete_DAG_generic_external_errors_cancellation_recovery_passed" if all(checks.values()) else "failed_DAG_supplement_guard",
            "checks": checks, "records": records, "sources": sources, "inputs": inputs, "complete_entry_seconds": time.perf_counter() - STARTED,
            "new_optimization_count": 0, "external_public_component_not_GT_unseen_whole_pipeline_or_training": True}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "seconds": result["complete_entry_seconds"]}), flush=True)
        if not all(checks.values()):
            raise AssertionError("An original DAG lifecycle or complete output protection failed")
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "failed_DAG_supplements_no_acceptance", "error": repr(error), "checks_before_failure": checks,
            "records": records, "sources": sources, "inputs": inputs, "complete_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise
    finally:
        native.main, native.projection_validity, sys.argv = original_main, original_projection, original_argv


if __name__ == "__main__":
    main()

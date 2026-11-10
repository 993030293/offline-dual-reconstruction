"""Check complete current Gaussian files and complete observed image metric inputs."""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import activate_reference, atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    runtime = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
    import torch
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import PILtoTorch
    from utils.loss_utils import create_window, ssim
    from vectorized_gaussian_read import vectorized_gaussian_read
    from ply_serialization import vectorized_ply_model
    from streamed_gaussian_ply_oracle_002 import streamed_ply_model
    from device_artifact_picture import DeviceArtifactPicture, native_picture
    from reference_ssim import ReferenceSSIM
    from interior_ssim_reuse import InteriorSSIM
    from fused_ssim_formula_v2 import FusedInteriorSSIM
    torch.set_num_threads(1)
    sources = [Path(__file__), V2 / "vectorized_gaussian_read.py", V1 / "ply_serialization.py",
        V3 / "streamed_gaussian_ply_oracle_002.py", V3 / "device_artifact_picture.py", V2 / "reference_ssim.py",
        V3 / "interior_ssim_reuse.py", V3 / "fused_ssim_formula.py", V3 / "fused_ssim_formula_v2.py"]
    source_ids = [data_identity(path) for path in sources]
    model_paths = {}
    for mode in ("baseline", "optimized"):
        side = args.campaign / args.case / mode
        trained = read(side / "training/STATE.json")
        if trained["status"] != "completed" or trained["iteration"] != trained["target_iterations"]:
            raise ValueError("The complete current training schedule is required")
        model_paths[mode] = side / f"training/model/point_cloud/iteration_{trained['iteration']}/point_cloud.ply"
    identities = [data_identity(path) for path in model_paths.values()]
    atomic_json(args.output / "CONTRACT.json", {"sources": source_ids, "models": identities, "runtime": runtime,
        "native_population_and_all_Gaussian_attributes": True, "image_metric_tolerance": 1e-6,
        "saved_PNG_metric_input_is_quantized_not_primary_float_render": True,
        "complete_evaluation_denominator": 24, "source_and_primary_results_unchanged": True,
        "CUDA_budget_bytes": 12*1024**3, "standalone_module_cost_not_whole_pipeline_gain": True})
    result = {"case": args.case, "families": {}, "status": "running"}

    def publish():
        atomic_json(args.output / "STATE.json", result)

    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    def close():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        reader_rows = []
        tensor_names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
        for mode, model_path in model_paths.items():
            for trial in range(3):
                models, durations = {}, {}
                order = ("native", "vector") if trial != 1 else ("vector", "native")
                for label in order:
                    before = clock()
                    with vectorized_gaussian_read(label == "vector"):
                        models[label] = GaussianModel(3)
                        models[label].load_ply(str(model_path))
                    durations[label] = clock()-before
                checks = {name: torch.equal(getattr(models["native"], name), getattr(models["vector"], name)) for name in tensor_names}
                reader_rows.append({"mode": mode, "trial": trial, "seconds": durations, "complete_attribute_exact": checks,
                    "vertices": len(models["native"]._xyz), "active_SH_degree_exact": models["native"].active_sh_degree == models["vector"].active_sh_degree})
                del models
                close()
        result["families"]["Q61"] = {"rows": reader_rows, "all_complete_attributes_exact": all(all(row["complete_attribute_exact"].values()) and row["active_SH_degree_exact"] for row in reader_rows)}
        publish()
        gaussian = GaussianModel(3)
        with vectorized_gaussian_read():
            gaussian.load_ply(str(model_paths["optimized"]))
        serialization_rows = []
        for trial in range(3):
            for label in (("native_oracle", "streamed_oracle") if trial != 1 else ("streamed_oracle", "native_oracle")):
                folder = args.output / (str(trial)+"_"+label)
                folder.mkdir()
                cls = vectorized_ply_model(GaussianModel, True) if label == "native_oracle" else streamed_ply_model(GaussianModel)
                model = cls(3)
                for name in tensor_names:
                    setattr(model, name, getattr(gaussian, name))
                before = clock()
                model.save_ply(str(folder / "point_cloud.ply"))
                seconds = clock()-before
                expected, actual = data_identity(model_paths["optimized"]), data_identity(folder / "point_cloud.ply")
                serialization_rows.append({"trial": trial, "label": label, "seconds": seconds,
                    "complete_current_source_bytes_exact": expected["sha256"] == actual["sha256"] and expected["bytes"] == actual["bytes"],
                    "event": model.ply_serialization_events, "population": len(model._xyz)})
                del model
        result["families"]["R127"] = {"rows": serialization_rows,
            "all_complete_source_bytes_exact": all(row["complete_current_source_bytes_exact"] for row in serialization_rows)}
        del gaussian
        close()
        publish()
        cases = {}
        for mode in ("baseline", "optimized"):
            location = read(args.campaign / args.case / mode / "localization/LOCALIZATION.json")
            cases[mode] = {row["image_name"]: row for row in location["rows"]}
        names = [row["image_name"] for row in read(args.campaign / args.case / "optimized/initial/EVALUATION_FRAMES.json")["rows"]]
        images, failed_rows = [], []
        from render_evaluation import mounted
        for name in names:
            rows = [cases[mode][name] for mode in ("baseline", "optimized")]
            if any(row["status"] != "localized" for row in rows):
                failed_rows.append({"image_name": name, "status": [row["status"] for row in rows]})
                continue
            reference_paths = [mounted(row["image_reference"]) for row in rows]
            if reference_paths[0].read_bytes() != reference_paths[1].read_bytes():
                raise ValueError("The complete common raw reference PNG changed")
            predictions = []
            for mode in ("baseline", "optimized"):
                with Image.open(args.campaign / args.case / mode / "evaluation" / (Path(name).stem+".png")) as photo:
                    array = np.asarray(photo.convert("RGB"), dtype=np.float32).copy()/255
                predictions.append(torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).contiguous().cuda())
            height, width = predictions[0].shape[-2:]
            with Image.open(reference_paths[0]) as photo:
                reference = PILtoTorch(photo.convert("RGB"), (width, height)).unsqueeze(0).cuda()
            images.append({"name": name, "prediction": predictions, "reference": reference,
                "crop": (round(height*.25), round(height*.75), round(width*.25), round(width*.75))})
        result["image_denominator"] = {"all_count": len(names), "common_rendered": len(images), "failures": failed_rows,
                                      "partial_inputs_do_not_accept_overall_quality": bool(failed_rows)}
        if not images:
            raise ValueError("No common current Gaussian render is available for a metric mechanism")
        picture_rows = []
        for trial in range(3):
            values = {}
            order = ("native", "device") if trial != 1 else ("device", "native")
            for label in order:
                adapter = DeviceArtifactPicture(True)
                before = clock()
                outputs = [native_picture(pred[0]) if label == "native" else adapter(pred[0]) for row in images for pred in row["prediction"]]
                seconds = clock()-before
                values[label] = {"outputs": outputs, "seconds": seconds, "trace": adapter.calls if label == "device" else None}
            picture_rows.append({"trial": trial, "seconds": {key: value["seconds"] for key, value in values.items()},
                "all_complete_pixels_exact": all(a.tobytes() == b.tobytes() for a, b in zip(values["native"]["outputs"], values["device"]["outputs"])),
                "device_trace": values["device"]["trace"]})
        result["families"]["R100"] = {"rows": picture_rows, "all_complete_pixels_exact": all(row["all_complete_pixels_exact"] for row in picture_rows)}
        publish()
        metric_rows, baseline_values = [], None
        for trial in range(3):
            choices = ("native", "reference", "view_major", "interior", "fused")
            for label in choices if trial != 1 else tuple(reversed(choices)):
                constructor = {"reference": ReferenceSSIM, "view_major": ReferenceSSIM, "interior": InteriorSSIM, "fused": FusedInteriorSSIM}.get(label)
                cache = constructor(create_window, 1024**3) if constructor else None
                operation = cache if cache else ssim
                values = {}
                before = clock()
                sequence = [(model, index) for model in range(2) for index in range(len(images))] if label != "view_major" else [(model, index) for index in range(len(images)) for model in range(2)]
                for model, index in sequence:
                    row = images[index]
                    prediction, reference = row["prediction"][model], row["reference"]
                    y1, y2, x1, x2 = row["crop"]
                    with torch.no_grad():
                        values[str(model)+"/"+row["name"]] = [float(operation(prediction, reference).item()),
                            float(operation(prediction[:, :, y1:y2, x1:x2], reference[:, :, y1:y2, x1:x2]).item())]
                    if label == "view_major" and model == 1:
                        cache.clear()
                seconds = clock()-before
                if baseline_values is None and label == "native":
                    baseline_values = values
                entry = {"trial": trial, "label": label, "seconds": seconds, "all_complete_scalar_values": values,
                         "trace": cache.receipt() if cache else None}
                metric_rows.append(entry)
                if cache:
                    cache.clear()
        if baseline_values is None:
            raise ValueError("A complete native SSIM reference is absent")
        for row in metric_rows:
            differences = [abs(value-baseline_values[key][index]) for key, values in row["all_complete_scalar_values"].items() for index, value in enumerate(values)]
            row["maximum_scalar_absolute_difference"] = max(differences)
            row["all_values_within_frozen_tolerance"] = max(differences) <= 1e-6
        result["families"]["Q30_R135_R134_R172"] = {"rows": metric_rows,
            "all_complete_common_image_values_within_tolerance": all(row["all_values_within_frozen_tolerance"] for row in metric_rows),
            "native_primary_float_metrics_not_overwritten": True, "fused_compile_and_cold_setup_in_first_trial_cost": True}
        del images
        close()
        result.update(status="complete_current_model_and_image_module_outcomes_recorded", elapsed_seconds=time.perf_counter()-started,
            sources_unchanged=source_ids == [data_identity(path) for path in sources],
            models_unchanged=identities == [data_identity(path) for path in model_paths.values()],
            maximum_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())
        if result["maximum_CUDA_reserved_bytes"] > 12*1024**3:
            raise MemoryError("The GPU module campaign exceeded its frozen CUDA allocation budget")
        atomic_json(args.output / "RESULT.json", result)
        publish()
    except BaseException as error:
        result.update(status="failed_current_module_attempt", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output / "FAILURE.json", result)
        raise


if __name__ == "__main__":
    main()

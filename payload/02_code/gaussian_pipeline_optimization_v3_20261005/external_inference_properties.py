"""Check native getter equivalence on synthetic parameters from all public photos."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
IMAGES = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
sys.path[:0] = [str(CODE), str(V1)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(IMAGES.glob("*.png"))
    if len(paths) != 360:
        raise ValueError("The original public photograph denominator changed")
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=300, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        from PIL import Image
        import torch
        from scene.gaussian_model import GaussianModel
        import scene.gaussian_model as gaussian_module
        from inference_property_workspace import FIELDS, PROPERTIES, InferencePropertyWorkspace
        sources = [data_identity(path) for path in (Path(__file__), CODE / "inference_property_workspace.py", Path(gaussian_module.__file__))]
        identities = [data_identity(path) for path in paths]
        copies = args.output / "source_copies"
        copies.mkdir()
        for index, row in enumerate(sources):
            shutil.copyfile(row["path"], copies / (f"{index:02d}_" + Path(row["path"]).name))
        atomic_json(args.output / "CONTRACT.json", {"sources": sources, "images": identities, "denominator": 360,
            "component_only_synthetic_native_parameters_not_trained_Oxford_Gaussians": True,
            "construction_before_results": "uniformly sampled <=129 RGB triples; SH DC from RGB/255; zero SH rest; deterministic colour-derived rotation, scaling, opacity, and synthetic XYZ",
            "strict_blind_test": False, "geometry_or_pose_GT_claim": False})
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()]

        def bits(value):
            return value.detach().contiguous().cpu().numpy().tobytes()

        rows = []
        for path in paths:
            with Image.open(path) as photo:
                colours = np.asarray(photo.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
            n = min(129, len(colours))
            values = torch.from_numpy(colours[np.linspace(0, len(colours) - 1, n, dtype=np.int64)].copy()).to("cuda", dtype=torch.float32) / 255
            owner = InferencePropertyWorkspace(GaussianModel, True)
            model = owner.model_class(3)
            constructed = {"_xyz": values - .5, "_features_dc": values[:, None, :],
                "_features_rest": torch.zeros((n, 15, 3), device="cuda"), "_opacity": values[:, :1] - .5,
                "_scaling": values - 1., "_rotation": torch.cat((torch.ones((n, 1), device="cuda"), values), dim=1)}
            for name, value in constructed.items():
                setattr(model, name, torch.nn.Parameter(value, requires_grad=True))
            before = {name: (bits(getattr(model, name)), getattr(model, name)._version) for name in FIELDS}
            with torch.no_grad():
                equal = all(bits(getattr(model, name)) == bits(owner.descriptors[name].fget(model))
                    and bits(getattr(model, name)) == bits(owner.descriptors[name].fget(model)) for name in PROPERTIES)
            intact = all((bits(getattr(model, name)), getattr(model, name)._version) == before[name] for name in FIELDS)
            owner.close()
            row = {"image": path.name, "rows": n, "all_four_properties_twice_native_bits_exact": equal,
                "all_synthetic_parameters_bits_versions_intact": intact,
                "actual_one_native_execution_and_one_hit_each": all(value == 1 for value in owner.trace["native_executions"].values())
                    and all(value == 1 for value in owner.trace["hits"].values()),
                "workspace_closed": owner.trace["closed"] and owner.resident_bytes == 0,
                "native_CUDA_elements": owner.trace["native_CUDA_elements"]}
            if not all(row[name] for name in ("all_four_properties_twice_native_bits_exact", "all_synthetic_parameters_bits_versions_intact",
                                              "actual_one_native_execution_and_one_hit_each", "workspace_closed")):
                raise AssertionError("An external colour-input native property contract changed")
            rows.append(row)
        checks = {"all360_original_photo_colour_inputs_checked": len(rows) == 360,
            "all_external_native_property_parameters_and_lifecycle_exact": all(row["all_four_properties_twice_native_bits_exact"]
                and row["all_synthetic_parameters_bits_versions_intact"] and row["actual_one_native_execution_and_one_hit_each"]
                and row["workspace_closed"] for row in rows),
            "all_original_images_and_native_sources_unchanged": identities == [data_identity(path) for path in paths]
                and sources == [data_identity(row["path"]) for row in sources],
            "native_CPU_and_all_CUDA_RNG_unchanged": bits(cpu_rng) == bits(torch.random.get_rng_state())
                and all(bits(a) == bits(b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())),
            "native_Gaussian_class_binding_unchanged": gaussian_module.GaussianModel is GaussianModel}
        result = {"status": "external360_component_passed" if all(checks.values()) else "failed_verification", "checks": checks,
            "rows": rows, "sources": sources, "inputs": identities, "complete_seconds": time.perf_counter() - STARTED,
            "scope": "synthetic_property_math_component_from_existing_nonblind360_RGB_inputs_no_trained_external_model_or_GT",
            "new_optimization_count": 0}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The public-input native getter component failed")


if __name__ == "__main__":
    main()

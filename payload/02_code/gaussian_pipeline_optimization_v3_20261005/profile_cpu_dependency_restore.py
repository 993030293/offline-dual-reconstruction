"""Compose an unchanged native restore with owned CPU dependency scheduling."""

from pathlib import Path

BASE = Path(__file__).with_name("profile_native_restore_cost.py")


def main():
    source = BASE.read_text(encoding="utf-8")
    replacements = [
        ('from private_checkpoint_load import private_checkpoint_load',
         'from private_checkpoint_load import private_checkpoint_load\nfrom native_cpu_dependency_bootstrap import native_cpu_dependency_bootstrap\nfrom fused_camera_digest import DigestLibrary\nimport hashlib'),
        ('    parser.add_argument("--output", type=Path, required=True)',
         '    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--mode", choices=("incumbent", "bootstrap"), required=True)'),
        ('    sources = [Path(__file__), CODE / "private_checkpoint_load.py", CODE / "mapped_image_linux_guard.py",',
         '    digest_library = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so"\n'
         '    native_Torch = Path("/opt/e26_thin_surface_v1/venv/lib/python3.11/site-packages/torch")\n'
         '    sources = [Path(__file__), CODE / "profile_native_restore_cost.py", CODE / "native_cpu_dependency_bootstrap.py",\n'
         '        CODE / "fused_camera_digest.py", digest_library, native_Torch / "__init__.py", native_Torch / "_compile.py",\n'
         '        native_Torch / "optim/optimizer.py", native_Torch / "_dynamo/__init__.py",\n'
         '        CODE / "private_checkpoint_load.py", CODE / "mapped_image_linux_guard.py",'),
        ('    runtime = activate_reference(RUNTIME)\n    import numpy as np', '    import numpy as np'),
        ('    identities = [data_identity(path) for path in inputs]',
         '    runtime = activate_reference(RUNTIME)\n'
         '    with LinuxRSSGuard(output, atomic_json, maximum_seconds=180, maximum_bytes=4 * 1024**3) as CPU_guard:\n'
         '        with native_cpu_dependency_bootstrap(enabled=args.mode == "bootstrap") as dependency:\n'
         '            identities = [data_identity(path) for path in inputs]\n'
         '    atomic_json(output / "CPU_BOOTSTRAP_RESOURCE.json", CPU_guard.row)'),
        ('with LinuxRSSGuard(output, atomic_json, maximum_seconds=540, maximum_bytes=16 * 1024**3):',
         'with LinuxRSSGuard(output, atomic_json, maximum_seconds=max(1, 540 - (time.perf_counter() - STARTED)), maximum_bytes=16 * 1024**3):'),
        ('            native_parser = argparse.ArgumentParser()',
         '            Torch_defaults = {"dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(),\n'
         '                "interop_threads": torch.get_num_interop_threads(), "grad_enabled": torch.is_grad_enabled()}\n'
         '            native_parser = argparse.ArgumentParser()'),
        ('            peak = torch.cuda.max_memory_reserved()',
         '            fingerprint_started = time.perf_counter()\n'
         '            cameras = scene.getTrainCameras()\n'
         '            digests = DigestLibrary(digest_library).images([camera.original_image for camera in cameras], workers=4)\n'
         '            camera_rows = []\n'
         '            for camera, digest in zip(cameras, digests):\n'
         '                row = {"name": camera.image_name, "uid": camera.uid, "colmap_id": camera.colmap_id,\n'
         '                    "width": camera.image_width, "height": camera.image_height, "FoVx": camera.FoVx, "FoVy": camera.FoVy,\n'
         '                    "image_shape": list(camera.original_image.shape), "image_stride": list(camera.original_image.stride()),\n'
         '                    "image_dtype": str(camera.original_image.dtype), "complete_pixel_sha256": digest}\n'
         '                for name in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"):\n'
         '                    value = getattr(camera, name)\n'
         '                    row[name] = {"shape": list(value.shape), "dtype": str(value.dtype),\n'
         '                        "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}\n'
         '                camera_rows.append(row)\n'
         '            phases["complete1016_camera_pixel_matrix_identity_seconds"] = time.perf_counter() - fingerprint_started\n'
         '            del cameras, camera, value\n'
         '            peak = torch.cuda.max_memory_reserved()'),
        ('                "all_original_nine_restore_fields":',
         '                "owned_CPU_dependency_worker_closed": dependency["owned_worker_joined"],\n'
         '                "complete1016_camera_rows_measured": len(camera_rows) == 1016,\n'
         '                "all_original_nine_restore_fields":'),
        ('            result = {"status": "complete_native_restore_cost_diagnostic_passed", "checks": checks,',
         '            result = {"status": "complete_cpu_dependency_restore_passed", "checks": checks,\n'
         '                "mode": args.mode, "CPU_dependency": dependency, "Torch_defaults": Torch_defaults,\n'
         '                "complete_original_camera_rows": camera_rows,'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The frozen native restore source changed at: " + before)
        source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_CPU_restore"}
    exec(compile(source, "<source_bound_native_CPU_restore>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

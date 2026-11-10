"""Run a bounded R82 variant through the original complete 24-camera copy/render path."""

from pathlib import Path


def main():
    native_path = Path(__file__).with_name("profile_streamed_copy.py")
    source = native_path.read_text(encoding="utf-8")
    replacements = {
        "from streamed_camera_copy import CameraCopy": "from registered_camera_copy import RegisteredCameraCopy as CameraCopy",
        '    args.output.mkdir(parents=True, exist_ok=False)':
            '    args.output.resolve().relative_to((ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/registered_camera_copy_001").resolve())\n'
            '    args.output.mkdir(parents=True, exist_ok=False)\n'
            '    import psutil\n    process = psutil.Process()\n'
            '    atomic_json(args.output/"PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})\n'
            '    (args.output/"EXECUTED_PROFILE.py").write_text(_executed_source, encoding="utf-8")\n'
            '    torch_threads = 4\n',
        '    build = gsplat_adapter.activate_gsplat(args.build)':
            '    torch.set_num_threads(torch_threads)\n    build = gsplat_adapter.activate_gsplat(args.build)',
        '    identities = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("streamed_camera_copy.py"), args.model, args.localization)]':
            '    identity_paths = [Path(__file__), Path(__file__).with_name("streamed_camera_copy.py"), Path(__file__).with_name("registered_camera_copy.py"), Path(__file__).with_name("profile_registered_camera_copy.py"), args.model, args.localization]\n'
            '    identity_paths += [mounted_path(row["image_reference"]) for row in localization["rows"]]\n'
            '    identities = [data_identity(path) for path in identity_paths]',
        '"minimum_complete_makespan_reduction": .1': '"minimum_complete_makespan_reduction": .05',
        '"single_slot_pinned_and_single_copy_stream_only": True':
            '"single_slot_pinned_and_single_copy_stream_only": True, "steps_each_side": 216, "R82_same_family_variant": True, "no_duplicate_CPU_staging": True, "bounded_stage_not_full_training": True',
        '(["native", "nonblocking", "streamed"] if trial % 2 == 0 else ["streamed", "nonblocking", "native"])':
            '(["native", "registered"] if trial % 2 == 0 else ["registered", "native"])',
        '            transfer = CameraCopy(mode)': '            transfer = CameraCopy(mode, [camera.original_image for camera in cameras])',
        '                for camera in cameras:': '                for step in range(216):\n                    camera = cameras[step % 24]',
        'for mode in ("nonblocking", "streamed")': 'for mode in ("registered",)',
        'for mode in ("native", "nonblocking", "streamed")': 'for mode in ("native", "registered")',
        '>= .1 else': '>= .05 else',
        '                    if torch.cuda.max_memory_reserved() > 12*1024**3:':
            '                    if (torch.cuda.max_memory_reserved() > 12*1024**3 or torch.cuda.mem_get_info()[0] < 2*1024**3 or process.memory_info().rss > 32*1024**3):',
        '    unchanged = identities == [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("streamed_camera_copy.py"), args.model, args.localization)]':
            '    unchanged = identities == [data_identity(path) for path in identity_paths]',
    }
    for before, after in replacements.items():
        expected = 2 if before == 'for mode in ("nonblocking", "streamed")' else 1
        if source.count(before) != expected:
            raise RuntimeError("The registered original profile source anchor changed: " + before)
        source = source.replace(before, after)
    namespace = {"__name__": "registered_copy_profile", "__file__": str(native_path), "_executed_source": source}
    exec(compile(source, str(native_path)+"#R82_registered_variant", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

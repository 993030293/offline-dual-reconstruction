"""Retain all original 43 complete pixel fixtures for the source-pinned native DLL."""

import argparse
from pathlib import Path
import sys

from native_camera_sampling import NativeCameraPixelRemap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    original = Path(__file__).with_name("check_camera_pixel_remap.py")
    source = original.read_text(encoding="utf-8")
    replacements = {
        'args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm001")':
            'args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001")',
        '    operator = CameraPixelRemap(original)': '    operator = _native_factory(original)',
        '    sources = [Path(__file__), Path(__file__).with_name("camera_pixel_remap.py")]':
            '    sources = [Path(__file__), Path(__file__).with_name("camera_pixel_remap.py"), Path(__file__).with_name("native_camera_sampling.py"), Path(__file__).with_name("native_camera_sampling.cpp"), Path(__file__).with_name("check_native_camera_sampling.py"), _library]',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("The original exact fixture source anchor changed: " + before)
        source = source.replace(before, after)
    namespace = {"__name__": "native_camera_sampling_generic", "__file__": str(original), "_library": args.library,
        "_native_factory": lambda original: NativeCameraPixelRemap(original, args.library)}
    exec(compile(source, str(original)+"#native_sampling", "exec"), namespace)
    previous = sys.argv
    try:
        sys.argv = [str(original), *remaining]
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

"""Build the strict channel-SIMD sampler using the existing owned compiler path."""

import argparse
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).with_name("build_native_quality_cache_writer.py")
    source = original.read_text(encoding="utf-8")
    before = source
    if source.count("native_quality_cache_writer") != 5:
        raise ValueError("The existing compiler and strict telemetry ownership anchors changed")
    source = source.replace("native_quality_cache_writer_001", "ud001/v5_simd_001")
    source = source.replace("native_quality_cache_writer", "native_vector_camera_sampling")
    anchor = '[str(compiler), "/nologo", "/LD", "/O2", "/MT", "/EHsc", "/std:c++17", "/W4",'
    if source.count(anchor) != 1:
        raise ValueError("The existing compiler argument contract changed")
    source = source.replace(anchor, '[str(compiler), "/nologo", "/LD", "/O2", "/MT", "/arch:AVX2", "/fp:strict", "/EHsc", "/std:c++17", "/W4",')
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    output.relative_to(root / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001")
    if output.exists():
        raise FileExistsError("A build output must be fresh")
    source = source.replace('    output.mkdir(parents=True, exist_ok=False)',
        '    output.mkdir(parents=True, exist_ok=False)\n'
        '    (output / "COMPLETE_EXECUTED_BUILD.py").write_text(_executed_build, encoding="utf-8")\n'
        '    (output / "BUILD_ADAPTER.diff").write_text(_build_diff, encoding="utf-8")')
    changes = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
                                          fromfile=original.name, tofile="strict_vector_sampler_build"))
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "strict_vector_sampler_build",
                 "_executed_build": source, "_build_diff": changes}
    exec(compile(source, "<existing_compiler_strict_vector_sampler>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()

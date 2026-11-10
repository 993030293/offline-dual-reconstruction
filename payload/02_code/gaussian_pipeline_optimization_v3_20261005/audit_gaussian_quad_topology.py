"""Check the existing indexed Gaussian quad and the draw-counter meaning."""

import argparse
import json
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/qt001")
    output.mkdir(parents=True, exist_ok=False)
    release = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release"
    paths = [Path(__file__), release / "shared-build/vendor/spark.module.js",
             release / "baseline/cases/packed/index.html", release / "baseline/cases/packed/viewer.js",
             release / "baseline/ASSET_MANIFEST.json", Path(__file__).with_name("behind_camera_draw.js")]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        (snapshot / (str(index) + "_" + path.name)).write_bytes(path.read_bytes())
    texts = [path.read_text(encoding="utf-8") for path in paths]
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})

    def array(name, kind):
        matches = list(re.finditer(r"var " + name + r" = new " + kind + r"\(\[([\s\S]*?)\]\);", texts[1]))
        if len(matches) != 1:
            raise ValueError("The actual quad array declaration changed")
        match = matches[0]
        values = [int(token.strip()) for token in match.group(1).split(",") if token.strip()]
        return values, texts[1][:match.start()].count("\n") + 1

    positions, positions_line = array("QUAD_VERTICES", "Float32Array")
    indices, indices_line = array("QUAD_INDICES", "Uint16Array")
    manifest = json.loads(texts[4])
    checks = {
        "exact_four_original_quad_positions": positions == [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0],
        "exact_six_indices_reusing_four_vertices": indices == [0, 1, 2, 0, 2, 3],
        "instanced_geometry_sets_original_position_and_index_buffers":
            'this.setAttribute("position", new THREE.BufferAttribute(QUAD_VERTICES, 3));' in texts[1]
            and 'this.setIndex(new THREE.BufferAttribute(QUAD_INDICES, 1));' in texts[1],
        "SparkRenderer_constructs_SplatGeometry": "const geometry = new SplatGeometry();" in texts[1],
        "packed_page_imports_shared_vendor":
            '"@sparkjsdev/spark": "../../../shared-build/vendor/spark.module.js"' in texts[2],
        "packed_viewer_constructs_SparkRenderer": "new SparkRenderer({" in texts[3]
            and 'from "@sparkjsdev/spark"' in texts[3],
        "actual_spark_bytes_match_both_asset_manifest_ids":
            sources[1]["sha256"] == manifest["scene"]["spark_sha256"] == manifest["box"]["spark_sha256"],
        "indexed_counter_vertices_are_submitted_indices_not_shader_invocations":
            'count = name.includes("Elements") ? args[1] : args[2]' in texts[5]
            and "trace.vertices += count * instances" in texts[5],
        "all_original_sources_unchanged": sources == [data_identity(path) for path in paths],
    }
    result = {"status": "already_indexed_quad_closed_count0" if all(checks.values()) else "failed_source_verification",
              "checks": checks, "sources": sources, "positions": positions, "indices": indices,
              "source_lines": {"positions": positions_line, "indices": indices_line,
                               "setIndex": 12144, "renderer_constructor": 12179, "counter": 114},
              "accepted_count_contribution": 0, "parent_accepted_count": 26,
              "no_GPU_or_shader_invocation_measurement": True, "original_triangle_metric_unchanged": True,
              "production_default_changed": False, "node_seconds": time.perf_counter() - started}
    atomic_json(output / "RESULT.json", result)
    if not all(checks.values()):
        raise ValueError("An original indexed quad source contract failed")
    print({key: result[key] for key in ("status", "checks", "accepted_count_contribution")}, flush=True)


if __name__ == "__main__":
    main()

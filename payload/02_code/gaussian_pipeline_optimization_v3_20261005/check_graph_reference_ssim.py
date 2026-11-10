"""Check owned graph execution against the unchanged complete SSIM consumer."""

import argparse
import inspect
from pathlib import Path
import shutil
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/eg001")
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from utils import loss_utils
        from interior_ssim_reuse import InteriorSSIM
        from graph_reference_ssim import GraphInteriorSSIM, _TAIL_SOURCE, _CALL_SOURCE
        from evaluate_interior_conv_work import finish_coupled_view

        paths = [Path(__file__), CODE / "graph_reference_ssim.py", CODE / "interior_ssim_reuse.py",
                 CODE / "view_major_evaluation.py", CODE / "evaluate_interior_conv_work.py",
                 ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/reference_ssim.py",
                 Path(loss_utils.__file__), Path(inspect.getfile(torch.cuda.CUDAGraph))]
        sources = [data_identity(path) for path in paths]
        for index, path in enumerate(paths):
            shutil.copyfile(path, args.output / (f"SOURCE_{index:02d}_" + path.name))
        (args.output / "ORIGINAL_TAIL.py").write_text(_TAIL_SOURCE, encoding="utf-8")
        (args.output / "COMPOSED_CALL.py").write_text(_CALL_SOURCE, encoding="utf-8")
        torch.cuda.reset_peak_memory_stats()
        rng = (torch.random.get_rng_state().tolist(), [value.tolist() for value in torch.cuda.get_rng_state_all()])
        rows, traces, close_checks = [], [], []
        with torch.no_grad():
            for shape in ((23, 37), (61, 73), (620, 812)):
                height, width = shape
                reference = torch.arange(height * width * 3, device="cuda", dtype=torch.float32).reshape(
                    1, height, width, 3).permute(0, 3, 1, 2) / (height * width * 3)
                native, graph = InteriorSSIM(loss_utils.create_window), GraphInteriorSSIM(loss_utils.create_window)
                for view in range(2):
                    current = reference if view == 0 else reference * .97 + .001
                    for model in range(2):
                        prediction = current.contiguous().flip(-1) * (.8 + .01 * model)
                        for region, left, right in (("full", prediction, current),
                                ("target", prediction[:, :, 2:-2, 3:-3], current[:, :, 2:-2, 3:-3])):
                            left_copy, right_copy = left.clone(), right.clone()
                            versions = left._version, right._version
                            expected = native(left, right).clone()
                            actual = graph(left, right).clone()
                            rows.append({"shape": list(shape), "view": view, "model": model, "region": region,
                                         "absolute_error": float((expected - actual).abs().item()),
                                         "inputs_unchanged": torch.equal(left, left_copy) and torch.equal(right, right_copy)
                                             and versions == (left._version, right._version)})
                    finish_coupled_view(native, [])
                    finish_coupled_view(graph, [])
                    graph.finish_reference()
                other = torch.ones((1, 3, height + 1, width + 1), device="cuda") * .3
                rows.append({"shape": list(shape), "region": "second_shape_native_before_capture",
                             "absolute_error": float((native(other, other) - graph(other, other)).abs().item()),
                             "inputs_unchanged": True})
                bad = torch.ones((1, 3, 11, 13), device="cuda", requires_grad=True)
                try:
                    graph(bad, bad.detach())
                except ValueError:
                    pass
                else:
                    raise AssertionError("The original training-use rejection did not propagate")
                traces.append(graph.receipt())
                native.clear()
                graph.clear()
                closed = graph.receipt()["owned_graph"]
                close_checks.append(closed["closed"] and closed["live_graph_entries"] == 0)
                try:
                    graph(other, other)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("The cleared graph accepted reuse")
        torch.cuda.synchronize()
        checks = {"all27_values_and_input_bits": len(rows) == 27 and all(
                      row["absolute_error"] <= 1e-7 and row["inputs_unchanged"] for row in rows),
                  "all_three_graphs_replayed_four_complete_full_maps": all(
                      value["owned_graph"]["captures"] == 1 and value["owned_graph"]["replays"] == 4 for value in traces),
                  "all_original_unsupported_second_shapes_native": all(
                      value["owned_graph"]["native_calls"] > 0 for value in traces),
                  "all_closed_graphs_reject_reuse": all(close_checks),
                  "all_sources_unchanged": sources == [data_identity(path) for path in paths],
                  "all_Torch_RNG_unchanged": rng == (torch.random.get_rng_state().tolist(),
                      [value.tolist() for value in torch.cuda.get_rng_state_all()]),
                  "actual_CUDA_budget_and_headroom": torch.cuda.max_memory_reserved() <= 12 * 1024**3
                      and torch.cuda.mem_get_info()[0] >= 2 * 1024**3}
        atomic_json(args.output / "RESULT.json", {"status": "generic_passed" if all(checks.values()) else "failed",
            "checks": checks, "rows": rows, "traces": traces, "sources": sources,
            "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("A graph compatibility or ownership guard failed")
        print("generic_passed", flush=True)


if __name__ == "__main__":
    main()

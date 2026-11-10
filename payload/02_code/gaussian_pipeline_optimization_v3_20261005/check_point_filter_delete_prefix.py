"""Force malformed native tracks through the original sequential deletion path."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import pycolmap

from check_point_filter_predicate import mutation, outcome
from point_filter_predicate import ROOT, V1, native, point_filter_execution
from point_filter_oracles import exact_values, native_state
from runtime import atomic_json, data_identity

OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/point_filter_predicate_001"
FIXTURE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/generic_binary_qa_001/curved_surface_1.0"


def structured(value):
    """Persist native object fields without losing types or floating-point bits."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return {"type": kind.__name__, "value": value}
    if kind is float:
        return {"type": "float", "bits": struct.pack("!d", value).hex()}
    if kind is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape), "hex": value.tobytes().hex()}
    if isinstance(value, np.generic):
        return {"type": kind.__name__, "dtype": value.dtype.str, "hex": value.tobytes().hex()}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[structured(key), structured(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {"type": kind.__name__, "items": [structured(item) for item in value]}
    if hasattr(value, "todict"):
        return {"type": kind.__module__ + "." + kind.__name__, "fields": structured(value.todict())}
    if hasattr(value, "value"):
        return {"type": kind.__module__ + "." + kind.__name__, "enum_value": int(value.value)}
    raise TypeError("The native state serializer does not support " + kind.__module__ + "." + kind.__name__)


def full_state(model, folder):
    state = native_state(model, folder)
    state["rig_fields"] = [(int(key), value.todict()) for key, value in model.rigs.items()]
    state["frame_fields"] = [(int(key), value.todict()) for key, value in model.frames.items()]
    state["image_cached_counts"] = [(int(key), int(image.num_points2D() if callable(image.num_points2D) else image.num_points2D),
        int(image.num_points3D() if callable(image.num_points3D) else image.num_points3D)) for key, image in model.images.items()]
    queries = ("num_cameras", "num_frames", "num_images", "num_points3D", "num_reg_frames", "num_reg_images", "num_rigs",
               "compute_num_observations", "compute_mean_track_length", "compute_mean_reprojection_error", "compute_mean_observations_per_reg_image")
    state["native_aggregate_queries"] = [(name, outcome(getattr(model, name))) for name in queries]
    return structured(state), state["native_binary_write"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWNED)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in
        ("point_filter_predicate.py", "point_filter_oracles.py", "check_point_filter_predicate.py")),
        Path(native.__file__), V1 / "runtime.py", Path(pycolmap._core.__file__)]
    paths = sources + sorted(FIXTURE.glob("*.bin"))
    identities = [data_identity(path) for path in paths]
    cases = ("duplicate_track", "inconsistent_point2D", "shared_track_element")
    policy = {"maximum_point_reprojection_error_px": 1e200, "minimum_point_track_length": 100}
    atomic_json(output / "CONTRACT.json", {"sources_and_inputs": identities, "cases": list(cases),
                "generic_policy_only": policy, "production_policy_unchanged": True,
                "purpose": "force_all_malformed_tracks_into_original_delete_not_only_update",
                "preserves_generic001_generic002_and_frozen_campaign": True, "counted": False})
    rows = []
    case_name, stage = "not_started", "setup"
    try:
        for case_name in cases:
            models = [pycolmap.Reconstruction(FIXTURE), pycolmap.Reconstruction(FIXTURE)]
            constructions = [outcome(lambda model=model: mutation(model, case_name)) for model in models]
            exact_values(*constructions)
            if not constructions[0]["ok"]:
                raise RuntimeError("Required forced-delete native fixture construction was rejected")
            initial, final, results, records = [], [], [], []
            for index, model in enumerate(models):
                label = "columns" if index else "native"
                stage = "snapshot_before"
                before, before_write = full_state(model, output / case_name / label / "before")
                initial.append(before)
                point_ids_before = list(model.points3D)
                stage = "actual_filter"
                with point_filter_execution(native, bool(index)) as trace:
                    calls = [outcome(lambda: native.filter_points(model, policy, np.array([-1., -1., -1.]), np.array([1., 1., 5.])))]
                    if calls[-1]["ok"]:
                        calls.append(outcome(lambda: native.filter_points(model, policy, np.array([-1., -1., -1.]), np.array([1., 1., 5.]))))
                stage = "snapshot_after"
                after, after_write = full_state(model, output / case_name / label / "after")
                final.append(after)
                results.append(calls)
                records.append({"label": label, "calls": calls, "trace": trace,
                    "point_ids_before": [int(point_id) for point_id in point_ids_before],
                    "point_ids_after": [int(point_id) for point_id in model.points3D],
                    "before_native_writer_outcome": before_write, "after_native_writer_outcome": after_write})
                atomic_json(output / case_name / label / "STATE_BITS.json", {"before": before, "after": after})
            stage = "compare_actual_semantics"
            exact_values(*initial)
            exact_values(*results)
            exact_values(*final)
            if not all(record["trace"]["original_binding_restored"] for record in records):
                raise AssertionError("The actual original binding was not restored")
            if not all(call["route"] == "columns" for call in records[1]["trace"]["calls"]):
                raise AssertionError("The forced-delete native case did not exercise the supported candidate")
            rows.append({"case": case_name, "records": records,
                         "all_native_point_image_camera_frame_rig_fields_and_cached_counts_exact": True,
                         "original_delete_error_type_message_and_mutation_prefix_exact": True})
            atomic_json(output / "STATE.json", {"status": "checking", "cases_complete": len(rows), "rows": rows, "counted": False})
        if [data_identity(path) for path in paths] != identities:
            raise RuntimeError("The frozen forced-delete sources or inputs changed")
        result = {"status": "forced_native_delete_prefix_oracle_passed", "cases": len(rows), "rows": rows,
                  "source_inputs_unchanged": True, "candidate_semantic_differences": 0, "counted": False}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "cases": len(rows)}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "candidate_semantics_failed" if stage == "compare_actual_semantics"
            else "validation_setup_or_entry_failed", "case": case_name, "stage": stage, "error_type": type(error).__name__,
            "error": str(error), "construction_outcomes": locals().get("constructions"),
            "records": locals().get("records"), "completed_rows": rows, "negative_preserved": True, "counted": False})
        raise


if __name__ == "__main__":
    main()

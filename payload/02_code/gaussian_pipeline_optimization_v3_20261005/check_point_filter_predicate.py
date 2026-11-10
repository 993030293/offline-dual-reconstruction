"""Stop R113 on the first local native outcome or mutation-prefix discrepancy."""

import argparse
import json
from pathlib import Path
import sys
import shutil
from types import SimpleNamespace

import numpy as np
import pycolmap

from point_filter_predicate import ROOT, V1, native, point_filter_execution
from point_filter_oracles import array_bits, exact_values, native_state
from runtime import atomic_json, data_identity

V3OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
POLICY = {"maximum_point_reprojection_error_px": 2., "minimum_point_track_length": 3}


def outcome(function):
    try:
        return {"ok": True, "value": function()}
    except BaseException as error:
        return {"ok": False, "type": type(error).__name__, "message": str(error)}


class MockTrack:
    def __init__(self, length, elements=()):
        self.size = length
        self.elements = list(elements)

    def length(self):
        return self.size


class MockModel:
    """Exercise the explicit original route for unsupported interfaces."""
    def __init__(self, failure=None):
        self.points3D = {1003: SimpleNamespace(xyz=np.array([20., 0., 0.]), error=5., track=MockTrack(0)),
                         7: SimpleNamespace(xyz=np.array([0., -0., 0.]), error=0., track=MockTrack(3)),
                         901: SimpleNamespace(xyz=np.array([30., 0., 0.]), error=0., track=MockTrack(2))}
        self.deleted = []
        self.updated = 0
        self.failure = failure

    def update_point_3d_errors(self):
        self.updated += 1
        if self.failure == "update":
            raise LookupError("native update fixture failed before deletion")

    def delete_point3D(self, point_id):
        if self.failure == "delete" and point_id == 901:
            raise IndexError("native delete fixture failed after first deletion")
        self.deleted.append(point_id)
        del self.points3D[point_id]


def mock_state(model):
    return {"updated": model.updated, "deleted": model.deleted,
            "points": [(key, array_bits(point.xyz), point.error, point.track.size,
                        [(element.image_id, element.point2D_idx) for element in point.track.elements])
                       for key, point in model.points3D.items()]}


def mutation(model, kind):
    ids = list(model.points3D)
    if not ids:
        return
    point = model.points3D[ids[0]]
    if kind.startswith("xyz_"):
        cases = {"xyz_nan": [np.nan, 0., 3.], "xyz_positive_inf": [np.inf, 0., 3.],
                 "xyz_negative_inf": [-np.inf, 0., 3.], "xyz_zero": [-0., 0., 3.],
                 "xyz_lower_equal": [-1., 0., 3.], "xyz_upper_equal": [1., 0., 3.],
                 "xyz_lower_outside": [np.nextafter(-1., -np.inf), 0., 3.],
                 "xyz_upper_outside": [np.nextafter(1., np.inf), 0., 3.]}
        point.xyz = np.array(cases[kind], dtype=np.float64)
    elif kind.startswith("track_length_"):
        length = int(kind.rsplit("_", 1)[1])
        while point.track.length() > length:
            point.track.delete_element(point.track.length() - 1)
    elif kind == "duplicate_track":
        element = point.track.elements[0]
        point.track.add_element(element.image_id, element.point2D_idx)
    elif kind == "inconsistent_point2D":
        element = point.track.elements[0]
        model.images[element.image_id].points2D[element.point2D_idx].point3D_id = ids[1]
    elif kind == "shared_track_element":
        element = point.track.elements[0]
        model.points3D[ids[1]].track.add_element(element.image_id, element.point2D_idx)
    elif kind == "dangling_image":
        point.track.add_element(2**31 - 17, 0)
    elif kind == "dangling_point2D":
        element = point.track.elements[0]
        point.track.add_element(element.image_id, 2**31 - 11)
    elif kind == "missing_camera_id":
        image = model.images[next(iter(model.images))]
        image.reset_camera_ptr()
        image.camera_id = 2**31 - 5
    elif kind != "unchanged":
        raise ValueError("An unknown required fixture mutation was requested")


def write_large_id_case(source, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("cameras.txt", "rigs.txt", "frames.txt"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    lines = (source / "points3D.txt").read_text(encoding="utf-8").splitlines()
    mapping = {int(line.split()[0]): int(line.split()[0]) + 2**53 + 10001
               for line in lines if line.strip() and not line.startswith("#")}
    rewritten = []
    for line in lines:
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            fields[0] = str(mapping[int(fields[0])])
            line = " ".join(fields)
        rewritten.append(line)
    (output / "points3D.txt").write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    lines = (source / "images.txt").read_text(encoding="utf-8").splitlines()
    rewritten = []
    observation = False
    for line in lines:
        if line.startswith("#"):
            rewritten.append(line)
            continue
        if observation:
            fields = line.split()
            for index in range(2, len(fields), 3):
                point_id = int(fields[index])
                if point_id >= 0:
                    fields[index] = str(mapping[point_id])
            line = " ".join(fields)
        rewritten.append(line)
        observation = not observation
    (output / "images.txt").write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    model = pycolmap.Reconstruction()
    model.read_text(output)
    model.write_binary(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(V3OUT / "point_filter_predicate_001")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("point_filter_predicate.py"),
               Path(__file__).with_name("point_filter_oracles.py"), Path(__file__).with_name("profile_point_filter_predicate.py"),
               Path(__file__).with_name("run_with_resource_lease.py"), Path(native.__file__), V1 / "runtime.py",
               Path(pycolmap.__file__), Path(pycolmap._core.__file__), Path(np._core._multiarray_umath.__file__)]
    fixtures = sorted(path for path in (V3OUT / "generic_binary_qa_001").glob("*/*.bin"))
    identities = [data_identity(path) for path in sources + fixtures]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(sources):
        if path.suffix == ".py":
            shutil.copyfile(path, snapshot / f"{index:02d}_{path.name}")
    atomic_json(output / "CONTRACT.json", {"sources_and_inputs": identities, "pycolmap": pycolmap.__version__,
                "numpy": np.__version__, "counted": False, "first_difference_stops": True})
    rows = []
    traces = []
    case_name = "not_started"
    try:
        folders = sorted({path.parent for path in fixtures})
        if len(folders) != 9:
            raise AssertionError("The frozen nine generic native binary fixtures are required")
        cases = [(folder.name + "_unchanged", folder, "unchanged", POLICY, False, True) for folder in folders]
        anchor = folders[0]
        case_name = "original_linked_camera_setter_rejection"
        models = [pycolmap.Reconstruction(anchor), pycolmap.Reconstruction(anchor)]
        rejected, states = [], []
        for index, model in enumerate(models):
            state_before = native_state(model, output / case_name / str(index) / "before")
            rejected.append(outcome(lambda model=model: setattr(model.images[next(iter(model.images))], "camera_id", 2**31 - 5)))
            state_after = native_state(model, output / case_name / str(index) / "after")
            exact_values(state_before, state_after)
            states.append(state_after)
        exact_values(*rejected)
        exact_values(*states)
        if rejected[0]["ok"]:
            raise AssertionError("The previously observed local linked-camera setter rejection changed")
        rows.append({"case": case_name, "construction_outcomes": rejected,
                     "rejected_constructor_state_prefix_exact_and_unchanged": True,
                     "filter_not_invoked_for_the_rejected_setup": True})
        atomic_json(output / "CONSTRUCTION_REJECTION.json", rows[-1])
        large_ids = write_large_id_case(anchor, output / "large_id_input")
        cases.append(("noncontiguous_IDs_above_float_exact_range", large_ids, "unchanged", POLICY, False, True))
        kinds = ["xyz_nan", "xyz_positive_inf", "xyz_negative_inf", "xyz_zero", "xyz_lower_equal",
                 "xyz_upper_equal", "xyz_lower_outside", "xyz_upper_outside", "track_length_0", "track_length_1",
                 "track_length_2", "track_length_3", "duplicate_track", "inconsistent_point2D", "shared_track_element",
                 "dangling_image", "dangling_point2D", "missing_camera_id"]
        cases += [(kind, anchor, kind, {**POLICY, "maximum_point_reprojection_error_px": 1e200}, False, True) for kind in kinds]
        cases += [("delete_all_then_empty_second_call", anchor, "unchanged", {**POLICY, "minimum_point_track_length": 100}, False, True),
                  ("all_three_reasons", anchor, "xyz_positive_inf", {**POLICY, "minimum_point_track_length": 100}, False, True),
                  ("frame_deregistration_rebuild", anchor, "unchanged", POLICY, True, True),
                  ("empty_model", None, "unchanged", POLICY, False, True),
                  ("native_missing_error_key", anchor, "unchanged", {"minimum_point_track_length": 3}, False, False),
                  ("native_missing_track_key", anchor, "unchanged", {"maximum_point_reprojection_error_px": 2.}, False, False),
                  ("native_nonfinite_threshold", anchor, "unchanged", {**POLICY, "maximum_point_reprojection_error_px": np.inf}, False, False),
                  ("native_oversized_integer_threshold", anchor, "unchanged", {**POLICY, "maximum_point_reprojection_error_px": 2**1024}, False, False)]
        for boundary in ("equal", "below", "above"):
            model = pycolmap.Reconstruction(anchor)
            model.update_point_3d_errors()
            threshold = float(model.points3D[next(iter(model.points3D))].error)
            threshold = (np.nextafter(threshold, -np.inf) if boundary == "below" else
                         np.nextafter(threshold, np.inf) if boundary == "above" else threshold)
            cases.append(("error_threshold_" + boundary, anchor, "unchanged",
                          {**POLICY, "maximum_point_reprojection_error_px": float(threshold)}, False, True))
        for case_name, folder, kind, policy, deregister, optimized in cases:
            models = [pycolmap.Reconstruction() if folder is None else pycolmap.Reconstruction(folder) for _ in range(2)]
            constructions = [outcome(lambda model=model: mutation(model, kind)) for model in models]
            atomic_json(output / "CURRENT_CONSTRUCTION.json", {"case": case_name, "outcomes": constructions})
            exact_values(*constructions)
            if not constructions[0]["ok"]:
                raise AssertionError("A required native fixture mutation is unsupported: " + case_name)
            lo, hi = np.array([-1., -1., -1.]), np.array([1., 1., 5.])
            outcomes, states = [], []
            for index, model in enumerate(models):
                calls = []
                with point_filter_execution(native, bool(index)) as trace:
                    calls.append(outcome(lambda: native.filter_points(model, policy, lo, hi)))
                    if calls[-1]["ok"]:
                        if deregister:
                            frame = next(iter(model.frames))
                            calls.append(outcome(lambda: model.deregister_frame(frame)))
                        if calls[-1]["ok"]:
                            calls.append(outcome(lambda: native.filter_points(model, policy, lo, hi)))
                if not trace["original_binding_restored"]:
                    raise AssertionError("The actual original binding was not restored")
                outcomes.append(calls)
                states.append(native_state(model, output / case_name / ("columns" if index else "native")))
                traces.append({"case": case_name, "enabled": bool(index), "trace": trace})
            exact_values(*outcomes)
            exact_values(*states)
            expected_route = "columns" if optimized else "original"
            if not all(call["route"] == expected_route for call in traces[-1]["trace"]["calls"]):
                raise AssertionError("A required native case used the wrong admitted route: " + case_name)
            rows.append({"case": case_name, "outcomes": outcomes[0], "all_order_types_bits_state_and_prefix_exact": True,
                         "optimized_native_route_exercised": optimized})
            atomic_json(output / "STATE.json", {"status": "checking", "completed_cases": len(rows), "latest": case_name, "counted": False})
        for failure in (None, "update", "delete"):
            for malformed in ("valid", "missing_error_key", "missing_track_key", "bad_extent", "bad_minimum"):
                case_name = "original_abstain_" + str(failure) + "_" + malformed
                policy = dict(POLICY)
                lo, hi = np.array([-1., -1., -1.]), np.array([1., 1., 5.])
                if malformed == "missing_error_key":
                    del policy["maximum_point_reprojection_error_px"]
                elif malformed == "missing_track_key":
                    del policy["minimum_point_track_length"]
                elif malformed == "bad_extent":
                    lo = np.zeros(2)
                elif malformed == "bad_minimum":
                    policy["minimum_point_track_length"] = "unsupported"
                models = [MockModel(failure), MockModel(failure)]
                outcomes, states = [], []
                for index, model in enumerate(models):
                    with point_filter_execution(native, bool(index)) as trace:
                        outcomes.append(outcome(lambda: native.filter_points(model, policy, lo, hi)))
                    states.append(mock_state(model))
                    if not trace["original_binding_restored"] or trace["calls"][0]["route"] != "original":
                        raise AssertionError("Unsupported input did not preserve the whole original route")
                exact_values(*outcomes)
                exact_values(*states)
                rows.append({"case": case_name, "outcomes": outcomes[0], "original_route_and_exception_prefix_exact": True,
                             "optimized_native_route_exercised": False})
        for error_type in (RuntimeError, KeyboardInterrupt):
            case_name = "context_restore_" + error_type.__name__
            previous = native.filter_points
            try:
                with point_filter_execution(native, True):
                    raise error_type("R113 restore fixture")
            except error_type:
                pass
            if native.filter_points is not previous:
                raise AssertionError("Context failure did not restore the original binding")
            rows.append({"case": case_name, "restore_exact": True})
        if [data_identity(Path(row["path"])) for row in identities] != identities:
            raise RuntimeError("Frozen oracle sources or fixture inputs changed")
        result = {"status": "generic_native_point_filter_oracle_passed", "cases": len(rows), "rows": rows, "traces": traces,
                  "all_return_types_order_float_bits_native_fields_and_error_prefixes_exact": True,
                  "source_and_inputs_unchanged": True, "counted": False}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "cases": len(rows)}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "first_required_oracle_failed", "case": case_name,
                    "error_type": type(error).__name__, "error": str(error), "completed_rows": rows,
                    "traces": traces, "construction_outcomes": locals().get("constructions"),
                    "negative_preserved": True, "counted": False})
        raise


if __name__ == "__main__":
    main()

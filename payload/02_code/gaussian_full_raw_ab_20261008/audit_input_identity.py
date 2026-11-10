"""Compare complete binary input identities and held Windows source protection."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V3), str(V3 / "parallel_b_005")]
from runtime import atomic_json, data_identity
from ordered_identity import collect_identities
from locked_input_identity import LockedInputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    admission = json.loads((args.side / "export/TRAINING_ADMISSION.json").read_text())
    dataset = Path(admission["dataset"])
    images = sorted((dataset / "images").glob("*.jpg"))
    if len(images) != admission["training_count"]:
        raise ValueError("Every current training image must enter the identity denominator")
    paths = [*images, *sorted((dataset / "sparse/0").glob("*.bin")), *sorted(Path(admission["training_opencv_model"]).glob("*.bin"))]
    source_paths = [Path(__file__), V3 / "parallel_b_005/ordered_identity.py", V3 / "locked_input_identity.py"]
    source_ids = [data_identity(path) for path in source_paths]
    original = [data_identity(path) for path in paths]
    rows = []
    try:
        for trial in range(3):
            for label in (("native", "ordered", "held") if trial != 1 else ("held", "ordered", "native")):
                before = time.perf_counter()
                trace = {}
                if label == "native":
                    values = [[data_identity(path) for path in paths] for _ in range(3)]
                elif label == "ordered":
                    values = [collect_identities(paths, data_identity, pass_trace) for pass_trace in [trace.setdefault(str(i), {}) for i in range(3)]]
                else:
                    owner = LockedInputs([args.side], data_identity)
                    try:
                        values = [[owner.identity(path) for path in paths] for _ in range(3)]
                        owner.verify()
                        trace = owner.receipt()
                    finally:
                        owner.close()
                seconds = time.perf_counter()-before
                identical = all(value == original for value in values)
                rows.append({"trial": trial, "label": label, "whole_three_pass_seconds_including_setup_verify_close": seconds,
                    "all_complete_identities_literal_exact": identical, "trace": trace})
                if not identical:
                    raise ValueError("The complete identity result or original ordering changed")
        copied = args.output / "protected_copy.jpg"
        shutil.copyfile(images[0], copied)
        owner = LockedInputs([args.output], data_identity)
        blocked = False
        try:
            owner.identity(copied)
            try:
                with copied.open("r+b") as stream:
                    stream.write(b"X")
            except PermissionError:
                blocked = True
            if not blocked:
                raise ValueError("The held Windows copy permitted byte mutation")
            owner.verify()
        finally:
            owner.close()
        with copied.open("r+b") as stream:
            stream.write(b"X")
        atomic_json(args.output / "RESULT.json", {"status": "complete_current_input_identity_outcomes_recorded",
            "rows": rows, "input_count": len(paths), "all_current_training_images": len(images),
            "sources": source_ids, "source_code_unchanged": source_ids == [data_identity(path) for path in source_paths],
            "primary_binary_inputs_unchanged": original == [data_identity(path) for path in paths],
            "isolated_Windows_write_blocked_while_held": blocked, "isolated_copy_mutable_after_close": True,
            "Windows_file_sharing_scope_only": True, "document_content_or_behavior_not_verified_by_hash": True,
            "elapsed_seconds": time.perf_counter()-started})
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_current_input_identity_audit",
            "rows": rows, "error": repr(error), "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()

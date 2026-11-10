"""Audit unchanged full SIMD export evidence before funding real recovery."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil

import numpy as np
import pycolmap

from profile_vector_full_export import BASE, LIBRARY, ROOT, read
from profile_sparse_ply_oracle import compare_outputs
from vector_camera_sampling import library_supports_simd
from runtime import atomic_json, data_identity

OWN = BASE / "v6_completion_001"


def main():
    output = OWN / "p0"
    output.mkdir(parents=True, exist_ok=False)
    identities, source_copies = {}, 0
    for ordinal in range(3):
        for mode in ("native", "vector"):
            folder = BASE / "c0" / ("t" + str(ordinal)) / mode
            contract = read(folder / "CONTRACT.json")
            for row in contract["sources"] + contract["inputs"]:
                if row["path"] in identities and identities[row["path"]] != row:
                    raise AssertionError("An original complete source or input differs across formal pairs")
                identities[row["path"]] = row
            for index, row in enumerate(contract["sources"]):
                if Path(row["path"]).suffix in (".py", ".cpp", ".json"):
                    saved = folder / "source_snapshot" / (f"{index:03d}_" + Path(row["path"]).name)
                    if data_identity(saved)["sha256"] != row["sha256"]:
                        raise AssertionError("An original executed source copy differs")
                    source_copies += 1
    if any(data_identity(row["path"]) != row for row in identities.values()):
        raise AssertionError("An original complete source or input changed")
    formal = read(BASE / "c0/RESULT.json")
    comparisons = [compare_outputs(BASE / "c0" / ("t" + str(ordinal)) / "native/export",
        BASE / "c0" / ("t" + str(ordinal)) / "vector/export") for ordinal in range(3)]
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks = {
        "original_three_full1050_outputs_still_exact": len(comparisons) == 3 and all(row["complete_byte_pairs"] == 1046
            and row["complete_JSON_pairs"] == 4 and row["complete_compared_bytes"] == 746975609 for row in comparisons),
        "all_six_formal_original_protections_passed": all(all(read(BASE / "c0" / ("t" + str(ordinal)) / mode / "RESULT.json")["checks"].values())
            for ordinal in range(3) for mode in ("native", "vector")),
        "original_generic_component_metadata_and_related_DAG_passed": all(all(read(BASE / name / "RESULT.json")["checks"].values())
            for name in ("g1", "g1_adapter", "s0", "m0", "d0")),
        "current_original_source_inputs_and_saved_copies_exact": True,
        "same_native_versions_and_actual_SIMD_admission": pycolmap.__version__ == "4.1.1" and np.__version__ == "2.3.5"
            and library_supports_simd(LIBRARY),
        "current32_active_and_R123_not_accepted": parent["additional_accepted_count"] == 32 and parent["status"].startswith("active")
            and not any(name.startswith("R123") for name in parent["accepted_mechanisms"]),
        "old_v0_never_executed_and_V5_failure_preserved": not (BASE / "v0").exists()
            and read(BASE / "FINAL_POST_EXIT_001.json")["new_optimization_count"] == 0,
        "original_compiled_CPP_source_copy_exact": (BASE / "b0/source_snapshot/native_vector_camera_sampling.cpp").read_bytes()
            == Path(__file__).with_name("native_vector_camera_sampling.cpp").read_bytes(),
    }
    old_cost = read(BASE / "FINAL_POST_EXIT_001.json")["known_fee_lower_bound_seconds"]
    medians = formal["medians"]
    delta = medians["native"]["phase_seconds"] - medians["vector"]["phase_seconds"]
    budget = (delta / .02 - medians["native"]["phase_seconds"]) * 6 - old_cost
    shutil.copyfile(__file__, output / Path(__file__).name)
    atomic_json(output / "CURRENT_IDENTITIES.json", list(identities.values()))
    atomic_json(output / "RESULT.json", {"status": "current_complete_evidence_reuse_verified" if all(checks.values()) else "failed_current_evidence_reuse",
        "checks": checks, "comparisons": comparisons, "current_source_input_identities": len(identities),
        "original_executed_source_copies": source_copies, "old_known_fee_lower_bound_seconds": old_cost,
        "additional_execution_budget_before_net2_boundary_seconds": budget, "current_versions": {
            "pycolmap": pycolmap.__version__, "numpy": np.__version__}, "new_optimization_count": 0,
        "evidence_is_observed_development_not_new_fresh_formal_or_blind": True,
        "actual_seconds_before_publication": time.perf_counter() - STARTED})
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    print(json.dumps({"checks": len(checks), "additional_seconds_budget": budget}), flush=True)


if __name__ == "__main__":
    main()

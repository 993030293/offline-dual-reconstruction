"""Check cache invalidation, interrupted publication, corruption and repeat benefit."""

import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

from quality_engine import ScoreCache, compact_quality, score_messages
from profile_quality import assert_scores
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    decision_path = project/"04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/frame_decisions.jsonl"
    rows = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if not row["excluded_role"]]
    selected = [rows[i] for i in np.linspace(0, len(rows)-1, 128, dtype=int)]
    ids = [row["camera_frame"] for row in selected]
    contract = {"source": data_identity(Path(__file__).with_name("quality_engine.py")),
                "opencv": cv2.__version__, "numpy": np.__version__, "revision": 1}
    atomic_json(args.output/"CONTRACT.json", {"frames": ids, "repetitions": 3, "maximum_seconds_per_trial": 300,
        "acceptance": "equal scores; source/version misses; partial files ignored; corrupt committed content rejected; hot median at least 10 percent faster",
        "cache_contract": contract})
    cache = ScoreCache(args.output/"cache", contract)
    cv2.setNumThreads(1)
    index = McapIndex(baseline.RAW)
    _, example = next(iter(index.iter_selected_messages(baseline.CAMERA_TOPIC, [ids[0]])))
    key = cache.key(example["payload"])
    modified = bytearray(example["payload"])
    modified[-1] ^= 1
    if cache.key(modified) == key:
        raise AssertionError("A changed image payload must miss its previous key")
    alternate = ScoreCache(args.output/"cache", {**contract, "revision": 2})
    if alternate.key(example["payload"]) == key:
        raise AssertionError("A changed implementation contract must miss its previous key")
    orphan = args.output/"cache"/(key+".interrupted.tmp")
    orphan.write_text('{"incomplete":', encoding="utf-8")
    if cache.read(key) is not None:
        raise AssertionError("An incomplete unpublished entry must not be reused")
    trials = []

    def run(label, active_cache):
        before = time.perf_counter()
        scores, timings = score_messages(index.iter_selected_messages(baseline.CAMERA_TOPIC, ids),
            decode_bayer, demosaic_rggb8_to_bgr, compact_quality, workers=4, capacity=16, cache=active_cache)
        elapsed = time.perf_counter()-before
        assert_scores(selected, scores)
        trials.append({"label": label, "seconds": elapsed, "cache_hits": timings["cache_hits"]})
        return scores

    cold = run("fill", cache)
    if trials[-1]["cache_hits"] != 0:
        raise AssertionError("First cache construction unexpectedly hit")
    if cache.read(cache.key(modified)) is not None or alternate.read(alternate.key(example["payload"])) is not None:
        raise AssertionError("Changed source/version reused a prior score")
    for repetition in range(3):
        for label in (["recompute", "hot"] if repetition != 1 else ["hot", "recompute"]):
            result = run(label, cache if label == "hot" else None)
            assert_scores(cold, result)
            if label == "hot" and trials[-1]["cache_hits"] != 128:
                raise AssertionError("Warm reuse failed to cover all frames")
    committed = args.output/"cache"/(key+".json")
    original = committed.read_text(encoding="utf-8")
    damaged = json.loads(original)
    damaged["quality"]["sharpness"] += 1
    committed.write_text(json.dumps(damaged), encoding="utf-8")
    rejected = False
    try:
        cache.read(key)
    except ValueError as error:
        rejected = "checksum" in str(error)
    if not rejected:
        raise AssertionError("A numerically plausible corrupt score was not rejected")
    committed.write_text(original, encoding="utf-8")
    cache.read(key)
    medians = {name: statistics.median(row["seconds"] for row in trials if row["label"] == name)
               for name in ["recompute", "hot"]}
    receipt = {"status": "passed_cache_contract", "trials": trials, "median_seconds": medians,
        "wall_reduction_fraction": 1-medians["hot"]/medians["recompute"],
        "source_and_contract_changes_miss": True, "partial_commit_ignored": True,
        "corrupt_value_rejected": True, "all_128_scores_equal": True,
        "full_sequence_integration_pending": True, "counted": False}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

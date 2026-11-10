"""Build object-independent incremental edges from time and measured covisibility."""

from collections import Counter
from bisect import bisect_left
import json


POLICY = {"temporal_each_side": 16, "covisibility_anchors": 2,
          "covisibility_neighbors_per_anchor": 12, "new_new": "all",
          "uses_object_class_or_ROI": False, "uses_GT": False,
          "selection_stage": "after_initial_SfM"}


def proposal_pairs(all_names, additions, by_name, previous, output=None):
    """Keep all images and old edges while limiting only new matching proposals."""
    new_names = set(additions)
    old_names = set(all_names) - new_names
    if not new_names or not old_names or set(all_names) != old_names | new_names:
        raise ValueError("An augmentation graph needs unique old and new images")
    if len(all_names) != len(set(all_names)) or any(name not in by_name for name in all_names):
        raise ValueError("Complete unique image timestamp metadata is required")
    registered = {image.name: image for image in previous.images.values() if image.has_pose}
    by_id = {int(image.image_id): image.name for image in previous.images.values()}
    ordered_old = sorted(old_names, key=lambda name: (by_name[name]["camera_header_ns"], name))
    stamps = [by_name[name]["camera_header_ns"] for name in ordered_old]
    covisibility_cache = {}

    def covisible(anchor):
        if anchor not in covisibility_cache:
            counts = Counter()
            image = registered[anchor]
            for point in image.points2D:
                if point.has_point3D():
                    landmark = previous.points3D[int(point.point3D_id)]
                    for element in landmark.track.elements:
                        name = by_id[int(element.image_id)]
                        if name in registered and name in old_names and name != anchor:
                            counts[name] += 1
            covisibility_cache[anchor] = sorted(counts, key=lambda name: (-counts[name], name))
        return covisibility_cache[anchor]

    pairs, decisions = set(), []

    def add(first, second):
        if first == second or not ({first, second} & new_names):
            raise AssertionError("New proposal edges must touch a new image")
        pairs.add(tuple(sorted((first, second))))

    for name in sorted(new_names):
        stamp = by_name[name]["camera_header_ns"]
        position = bisect_left(stamps, stamp)
        count = POLICY["temporal_each_side"]
        temporal = ordered_old[max(0, position-count):position] + ordered_old[position:position+count]
        anchors = sorted((item for item in temporal if item in registered),
                         key=lambda item: (abs(by_name[item]["camera_header_ns"]-stamp), item))[:POLICY["covisibility_anchors"]]
        neighbors = set(temporal)
        for anchor in anchors:
            neighbors.update(covisible(anchor)[:POLICY["covisibility_neighbors_per_anchor"]])
        for neighbor in neighbors:
            add(name, neighbor)
        decisions.append({"image_name": name, "temporal": temporal, "anchors": anchors,
                          "old_neighbors": sorted(neighbors)})
    ordered_new = sorted(new_names)
    for i, first in enumerate(ordered_new):
        for second in ordered_new[i+1:]:
            add(first, second)
    if any(not any(name in pair for pair in pairs) for name in new_names):
        raise AssertionError("Every new image must retain a matching opportunity")
    result = sorted(pairs)
    if output is not None:
        receipt = {"policy": POLICY, "old_count": len(old_names), "new_count": len(new_names),
                   "all_image_names_preserved": sorted(all_names), "decisions": decisions,
                   "proposed_new_pair_count": len(result),
                   "original_exhaustive_new_pair_count": len(new_names)*len(old_names)+len(new_names)*(len(new_names)-1)//2,
                   "old_pair_records_not_modified": True,
                   "no_coarse_pose_extrapolation_or_new_priors": True}
        (output / "INCREMENTAL_PAIR_POLICY.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result

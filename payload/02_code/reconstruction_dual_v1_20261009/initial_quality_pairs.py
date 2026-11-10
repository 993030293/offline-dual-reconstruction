"""Reuse the existing coarse SfM candidate graph before selection freeze."""

PAIR_BASE = 2147483647


def quality_pairs(candidate_pairs, selected, image_ids, attempted, temporal):
    ordered = sorted(selected, key=lambda row: (row['camera_header_ns'], row['camera_frame'], row['image_name']))
    planned, receipt = candidate_pairs(ordered, indexed=True)
    proposed = set()
    for first, second in planned:
        if not 0 <= first < second < len(ordered):
            raise ValueError('The existing coarse candidate graph has an invalid image index')
        proposed.add((ordered[first]['image_name'], ordered[second]['image_name']))
    identities = set()
    for first, second in proposed | set(temporal):
        if first == second or first not in image_ids or second not in image_ids:
            raise ValueError('A quality pair has no distinct actual extracted image identities')
        a, b = sorted((image_ids[first], image_ids[second]))
        if PAIR_BASE * a + b not in attempted:
            identities.add((a, b))
    names = {key: name for name, key in image_ids.items()}
    if len(names) != len(image_ids):
        raise ValueError('Quality pairing requires distinct native image identities')
    return [(names[a], names[b]) for a, b in sorted(identities)], {
        'existing_coarse_SfM_candidate_graph': receipt,
        'temporal_unattempted_pairs': len(temporal),
        'combined_unattempted_unique_pairs': len(identities),
        'coarse_poses_role': 'pair_proposals_only_actual_native_matching_and_geometry_still_required',
        'old_SfM_model_or_final_poses_used': False}

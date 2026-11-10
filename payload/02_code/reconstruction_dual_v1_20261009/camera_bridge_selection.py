"""Select raw-quality bridge views under a global temporal coverage budget."""

from bisect import bisect_left
import math

BINS = [(-.8, -.4), (-.4, -.15), (-.15, 0), (0, .15), (.15, .4), (.4, .8)]


def select_bridges(rows, original_names, triggers, maximum_new_images=64, *, critical_triggers=()):
    if isinstance(maximum_new_images, bool) or not isinstance(maximum_new_images, int) or not 1 <= maximum_new_images <= 64:
        raise ValueError('The existing bridge budget must be an integer in 1..64')
    lookup = {row['image_name']: row for row in rows}
    critical_triggers = set(critical_triggers)
    if not critical_triggers <= set(triggers) or not critical_triggers <= lookup.keys():
        raise ValueError('Required recovery priorities must identify actual trigger frames')
    candidates = sorted((row for row in rows if row['decision'] == 'eligible_not_selected'
        and row['coarse_pose_valid'] and not row['excluded_role'] and row['image_name'] not in original_names),
        key=lambda row: (row['camera_header_ns'], row['camera_frame'], row['image_name']))
    if any(not math.isfinite(row['quality']['quality_score']) for row in candidates):
        raise ValueError('An eligible raw bridge candidate has a nonfinite quality score')
    stamps = [row['camera_header_ns'] for row in candidates]
    needs, local_winners = [], set()
    need_masks = [0] * len(candidates)
    trigger_masks = [0] * len(candidates)

    def quality_order(index):
        row = candidates[index]
        return (-row['quality']['quality_score'], row['camera_frame'], row['image_name'])

    ordered_triggers = sorted(triggers)
    critical_trigger_mask = sum(1 << index for index, name in enumerate(ordered_triggers)
        if name in critical_triggers)
    for trigger_index, name in enumerate(ordered_triggers):
        stamp = lookup[name]['camera_header_ns']
        for low, high in BINS:
            first = bisect_left(stamps, stamp + round(low * 1e9))
            last = bisect_left(stamps, stamp + round(high * 1e9))
            eligible = list(range(first, last))
            best = min(eligible, key=quality_order) if eligible else None
            need_index = len(needs)
            needs.append((name, low, high, eligible, best))
            if best is not None:
                local_winners.add(best)
            for index in eligible:
                need_masks[index] |= 1 << need_index
                trigger_masks[index] |= 1 << trigger_index

    # Preserve the previous choice when its independent bin winners fit the budget.
    if len(local_winners) <= maximum_new_images:
        selected = set(local_winners)
        algorithm = 'legacy_six_bin_quality_winners_within_global_budget'
    else:
        selected, covered_need_mask, covered_trigger_mask = set(), 0, 0
        critical_need_mask = sum(1 << index for index, need in enumerate(needs)
            if need[0] in critical_triggers and need[3])
        # Cover required recovery bins before spending support views on observed weak poses.
        while critical_need_mask & ~covered_need_mask and len(selected) < maximum_new_images:
            available = [index for index, mask in enumerate(need_masks)
                if index not in selected and mask & critical_need_mask & ~covered_need_mask]
            if not available:
                break

            def required_coverage_order(index):
                return (-(trigger_masks[index] & critical_trigger_mask & ~covered_trigger_mask).bit_count(),
                    -(need_masks[index] & critical_need_mask & ~covered_need_mask).bit_count(),
                    *quality_order(index))

            best = min(available, key=required_coverage_order)
            selected.add(best)
            covered_need_mask |= need_masks[best]
            covered_trigger_mask |= trigger_masks[best]
        while len(selected) < maximum_new_images:
            available = [index for index, mask in enumerate(need_masks)
                if index not in selected and mask & ~covered_need_mask]
            if not available:
                break

            def coverage_order(index):
                # Cover previously unsupported cameras before adding more bins for covered ones.
                return (-(trigger_masks[index] & ~covered_trigger_mask).bit_count(),
                    -(need_masks[index] & ~covered_need_mask).bit_count(), *quality_order(index))

            best = min(available, key=coverage_order)
            selected.add(best)
            covered_need_mask |= need_masks[best]
            covered_trigger_mask |= trigger_masks[best]
        algorithm = ('required_recovery_bins_then_global_trigger_and_bin_coverage_quality_tiebreak'
            if critical_triggers else 'global_trigger_then_bin_coverage_quality_tiebreak')

    selection = []
    eligible_triggers, covered_triggers = set(), set()
    for name, low, high, eligible, local_best in needs:
        choices = [index for index in eligible if index in selected]
        chosen = min(choices, key=quality_order) if choices else None
        if eligible:
            eligible_triggers.add(name)
        if chosen is not None:
            covered_triggers.add(name)
        selection.append({'trigger_image': name, 'relative_time_bin_seconds': [low, high],
            'trigger_role': 'required_camera_recovery' if name in critical_triggers else 'observed_pose_support',
            'eligible_candidates': len(eligible),
            'local_quality_best': candidates[local_best]['image_name'] if local_best is not None else None,
            'chosen': candidates[chosen]['image_name'] if chosen is not None else None,
            'status': 'raw_quality_candidate_selected_geometry_pending' if chosen is not None
                else 'abstain_no_eligible_raw_candidate' if not eligible else 'abstain_bin_not_covered_within_budget'})
    additions = {candidates[index]['image_name']: candidates[index]
        for index in sorted(selected, key=lambda index: (candidates[index]['camera_frame'], candidates[index]['image_name']))}
    summary = {'algorithm': algorithm, 'eligible_candidate_pool': len(candidates),
        'independent_bin_winners_before_global_budget': len(local_winners),
        'maximum_new_images': maximum_new_images, 'selected_new_images': len(additions),
        'trigger_count': len(triggers), 'covered_trigger_count': len(covered_triggers),
        'critical_trigger_count': len(critical_triggers),
        'critical_triggers_without_eligible_candidates': sorted(critical_triggers - eligible_triggers),
        'eligible_critical_triggers_not_covered_within_budget': sorted(
            critical_triggers & eligible_triggers - covered_triggers),
        'uncovered_eligible_critical_bin_count': sum(row['trigger_image'] in critical_triggers
            and bool(row['eligible_candidates']) and row['chosen'] is None for row in selection),
        'triggers_without_eligible_raw_candidates': sorted(set(triggers) - eligible_triggers),
        'eligible_triggers_not_covered_within_budget': sorted(eligible_triggers - covered_triggers),
        'eligible_bin_count': sum(bool(row['eligible_candidates']) for row in selection),
        'selected_bin_count': sum(row['chosen'] is not None for row in selection),
        'uncovered_eligible_bin_count': sum(bool(row['eligible_candidates']) and row['chosen'] is None for row in selection),
        'original_registered_frames_removed_by_selection': 0,
        'selection_does_not_verify_geometry': True}
    return additions, selection, summary

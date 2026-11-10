"""Check bounded temporal coverage without depending on any object shape or recording."""

import random
import unittest

from camera_bridge_selection import select_bridges


def raw_row(index, seconds, score, decision='eligible_not_selected', coarse=True, excluded=False):
    stamp = round(seconds * 1e9)
    return {'image_name': f'frame_{index:05d}_{stamp}.jpg', 'camera_frame': index,
        'camera_header_ns': stamp, 'quality': {'quality_score': score},
        'decision': decision, 'coarse_pose_valid': coarse, 'excluded_role': excluded}


class BridgeSelectionTests(unittest.TestCase):
    def test_choices_within_budget_preserve_existing_bin_quality_winners(self):
        trigger = raw_row(100, 100, 1, 'selected_initial_sfm_pending')
        rows = [trigger, raw_row(1, 99.3, 1), raw_row(2, 99.7, 2),
            raw_row(3, 100.05, 3), raw_row(4, 100.10, 4)]
        additions, selection, coverage = select_bridges(rows, {trigger['image_name']}, {trigger['image_name']})
        self.assertEqual(set(additions), {rows[index]['image_name'] for index in [1, 2, 4]})
        self.assertEqual(coverage['algorithm'], 'legacy_six_bin_quality_winners_within_global_budget')
        self.assertTrue(all(row['chosen'] == row['local_quality_best'] for row in selection))

    def fixture(self):
        rows, triggers = [], set()
        for group, center in enumerate([100, 102, 104]):
            trigger = raw_row(1000 + group, center, 1, 'selected_initial_sfm_pending')
            rows.append(trigger)
            triggers.add(trigger['image_name'])
            for offset_index, offset in enumerate([-.7, -.3, -.05, .05, .3, .7]):
                score = (10 if group == 0 else 1) + offset_index * .01
                rows.append(raw_row(group * 10 + offset_index, center + offset, score))
        return rows, triggers

    def test_over_budget_prioritizes_uncovered_triggers_and_keeps_abstentions(self):
        rows, triggers = self.fixture()
        additions, selection, coverage = select_bridges(rows, triggers, triggers, maximum_new_images=3)
        self.assertEqual(len(additions), 3)
        self.assertEqual(coverage['independent_bin_winners_before_global_budget'], 18)
        self.assertEqual(coverage['covered_trigger_count'], 3)
        self.assertEqual(coverage['eligible_triggers_not_covered_within_budget'], [])
        self.assertGreater(coverage['uncovered_eligible_bin_count'], 0)
        self.assertTrue(any(row['status'] == 'abstain_bin_not_covered_within_budget' for row in selection))
        self.assertEqual(coverage['original_registered_frames_removed_by_selection'], 0)
        self.assertTrue(coverage['selection_does_not_verify_geometry'])

    def test_selection_is_deterministic_when_input_order_changes(self):
        rows, triggers = self.fixture()
        before = select_bridges(rows, triggers, triggers, maximum_new_images=3)
        random.Random(7).shuffle(rows)
        after = select_bridges(rows, triggers, set(reversed(sorted(triggers))), maximum_new_images=3)
        self.assertEqual(before, after)

    def test_original_raw_quality_coarse_and_role_eligibility_is_preserved(self):
        trigger = raw_row(100, 100, 1, 'selected_initial_sfm_pending')
        valid = raw_row(1, 100.05, 1)
        rows = [trigger, valid, raw_row(2, 100.06, 100, excluded=True),
            raw_row(3, 100.07, 100, coarse=False), raw_row(4, 100.08, 100, 'quality_rejected')]
        additions, _, _ = select_bridges(rows, {trigger['image_name']}, {trigger['image_name']})
        self.assertEqual(set(additions), {valid['image_name']})

    def test_temporal_bins_are_lower_inclusive_and_upper_exclusive(self):
        trigger = raw_row(100, 100, 1, 'selected_initial_sfm_pending')
        rows = [trigger, raw_row(1, 99.2, 1), raw_row(2, 99.6, 2),
            raw_row(3, 100, 3), raw_row(4, 100.8, 100)]
        additions, selection, _ = select_bridges(rows, {trigger['image_name']}, {trigger['image_name']})
        self.assertEqual(selection[0]['chosen'], rows[1]['image_name'])
        self.assertEqual(selection[1]['chosen'], rows[2]['image_name'])
        self.assertEqual(selection[2]['chosen'], None)
        self.assertEqual(selection[3]['chosen'], rows[3]['image_name'])
        self.assertNotIn(rows[4]['image_name'], additions)

    def test_no_eligible_candidates_cannot_fabricate_support(self):
        trigger = raw_row(100, 100, 1, 'selected_initial_sfm_pending')
        additions, selection, coverage = select_bridges([trigger], {trigger['image_name']}, {trigger['image_name']})
        self.assertEqual(additions, {})
        self.assertEqual(coverage['triggers_without_eligible_raw_candidates'], [trigger['image_name']])
        self.assertTrue(all(row['status'] == 'abstain_no_eligible_raw_candidate' for row in selection))

    def test_bridge_resource_budget_cannot_silently_grow(self):
        for budget in [0, 65, True, 1.5]:
            with self.assertRaises(ValueError):
                select_bridges([], set(), set(), budget)

    def test_required_recovery_receives_bins_before_observed_weak_poses(self):
        rows, triggers = self.fixture()
        required = rows[14]['image_name']
        additions, selection, coverage = select_bridges(rows, triggers, triggers,
            maximum_new_images=7, critical_triggers={required})
        selected_required = [row for row in selection if row['trigger_image'] == required and row['chosen']]
        self.assertEqual(len(selected_required), 6)
        self.assertEqual(coverage['uncovered_eligible_critical_bin_count'], 0)
        self.assertEqual(len(additions), 7)
        self.assertTrue(all(row['trigger_role'] == 'required_camera_recovery' for row in selected_required))

    def test_required_recovery_priority_does_not_increase_a_small_budget(self):
        rows, triggers = self.fixture()
        required = rows[14]['image_name']
        additions, selection, coverage = select_bridges(rows, triggers, triggers,
            maximum_new_images=3, critical_triggers={required})
        self.assertEqual(len(additions), 3)
        self.assertTrue(all(103.2 <= row['camera_header_ns'] / 1e9 < 104.8 for row in additions.values()))
        self.assertEqual(coverage['uncovered_eligible_critical_bin_count'], 3)
        self.assertTrue(coverage['selection_does_not_verify_geometry'])

    def test_required_priority_preserves_all_quality_winners_when_they_fit(self):
        rows, triggers = self.fixture()
        before = select_bridges(rows, triggers, triggers)
        after = select_bridges(rows, triggers, triggers, critical_triggers={rows[14]['image_name']})
        self.assertEqual(before[0], after[0])
        self.assertEqual(after[2]['algorithm'], 'legacy_six_bin_quality_winners_within_global_budget')

    def test_required_priority_must_be_an_actual_known_trigger(self):
        rows, triggers = self.fixture()
        for names in ({'unknown'}, {rows[1]['image_name']}):
            with self.assertRaisesRegex(ValueError, 'actual trigger frames'):
                select_bridges(rows, triggers, triggers, critical_triggers=names)


if __name__ == '__main__':
    unittest.main()

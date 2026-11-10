"""Use directed focus and retain every established ground, density, and ROI gate."""

import argparse
from pathlib import Path
import sys

from common import atomic_json, read_json
from metric_point_policy import native_point_policy
from target_focus import forward_focus
from target_prepare import PACKAGE

sys.path.insert(0, str(PACKAGE / 'code'))
import derive_vehicle_roi as legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric-sparse', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    _, _, renderer, near_m = native_point_policy()
    witness = {}

    def solve(centers, directions):
        try:
            focus, residuals, iterations, result = forward_focus(centers, directions, near_m)
        except ValueError as error:
            failure = args.output.with_name(args.output.stem + '_FOCUS_FAILURE.json')
            if failure.exists():
                raise FileExistsError('Do not replace an earlier focus failure witness') from error
            atomic_json(failure, {'status': 'failed_directed_focus_original_constraints_preserved',
                'metric_sparse': str(args.metric_sparse.resolve()), 'original_error': str(error),
                'actual_constraint_evidence': getattr(error, 'focus_constraint_evidence', None),
                'renderer_near_plane_m': near_m, 'camera_denominator_reduced': False,
                'camera_directions_flipped': False, 'manual_target_coordinates_used': False})
            raise
        witness.clear()
        witness.update(result)
        return focus, residuals, iterations

    original = legacy.solve_focus
    legacy.solve_focus = solve
    try:
        legacy.main()
    finally:
        legacy.solve_focus = original
    receipt = read_json(args.output)
    if (receipt['status'] != 'complete' or not witness.get('solver_success')
            or receipt['roi_derivation']['camera_count'] != witness['camera_count']
            or receipt['roi_derivation']['forward_depth_m']['positive_count'] != witness['actual_positive_count']):
        raise ValueError('The complete ROI receipt differs from its actual directed focus witness')
    witness['native_renderer_source'] = str(renderer)
    receipt['roi_derivation']['method'] = 'directed optical-ray focus, unchanged sparse floor and density support'
    receipt['roi_derivation']['forward_focus_fit'] = witness
    receipt['roi_derivation']['original_ground_density_and_ROI_gates_unchanged'] = True
    atomic_json(args.output, receipt)


if __name__ == '__main__':
    main()

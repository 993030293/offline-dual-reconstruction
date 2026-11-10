"""Bound sampled net gains with nonnegative unmetered equal-side fees."""

import math


def equal_fee_cost_bound(native_seconds, candidate_seconds, known_fee_seconds, sides=6):
    if (type(sides) is not int or sides < 1 or not all(math.isfinite(v) for v in (
            native_seconds, candidate_seconds, known_fee_seconds))
            or native_seconds <= 0 or candidate_seconds < 0 or known_fee_seconds < 0):
        raise ValueError("A finite positive reference and nonnegative bounded costs are required")
    native = native_seconds + known_fee_seconds / sides
    candidate = candidate_seconds + known_fee_seconds / sides
    signed = 1 - candidate / native
    # Extra equal fees move a negative gain toward zero, rather than downward.
    upper = max(0.0, signed)
    return {"native_lower_cost_seconds": native, "candidate_lower_cost_seconds": candidate,
            "signed_reduction_at_known_lower_fee": signed, "conservative_sample_reduction_upper_bound": upper}

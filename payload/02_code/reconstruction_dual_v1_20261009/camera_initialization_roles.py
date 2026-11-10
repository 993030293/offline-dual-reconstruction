"""Separate visual initializers, required recovery frames, and optional witnesses."""

VISUAL_RECOVERY_PROTOCOL = 'registered_weak_and_all_missing_initial_v3_indexed_fixed_map_retry_cache'
SENSOR_INITIALIZATION_PROTOCOL = 'genuine_visual_recovery_v11_missing_registration_feature_recovery_shared_budget_valid_tracks_joint_conditioning'
NATIVE_POINT_REBUILD_INPUT_STATUS = 'native_visual_registered_point_rebuild_REQUIRED'

VISUAL_SUPPORT_STATUSES = frozenset({
    'coarse_initialized_visual_support_ready_BA_REQUIRED',
    'native_visual_registered_support_ready_BA_REQUIRED',
    'native_point_rebuilt_visual_support_ready_BA_REQUIRED',
    'existing_point_policies_applied_BA_required',
    'native_joint_visual_conditioned_support_ready_metric_BA_REQUIRED',
})


def visual_recovery_targets(observations, initial_names, errors, minimum, conflict_m):
    """Include missing required cameras even when all registered poses look healthy."""
    registered, initial = set(observations), set(initial_names)
    weak = {name for name, count in observations.items() if count < minimum or errors.get(name, 0) > conflict_m}
    missing = initial - registered
    return {'registered_suspect_names': sorted(weak), 'missing_initial_names': sorted(missing),
        'target_names': sorted(weak | missing), 'required_names': sorted(registered | initial)}


def initialization_roles(observations, initial_names, additions, coarse_names, supported_observations=None,
                         unverified_conflicted_names=()):
    original = set(observations)
    initial_names, additions, coarse_names = map(set, (initial_names, additions, coarse_names))
    if additions & (original | initial_names):
        raise ValueError('A bridge candidate cannot replace an existing or initially selected frame')
    missing_initial = initial_names - original
    unobserved_original = {name for name, count in observations.items() if count == 0}
    if supported_observations is None:
        supported_observations = observations
    if set(supported_observations) != original or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > observations[name]
            for name, count in supported_observations.items()):
        raise ValueError('Qualified seed support must account for every actual registered camera')
    unsupported_original = {name for name, count in supported_observations.items() if count == 0}
    conflicted = set(unverified_conflicted_names)
    if not conflicted <= original:
        raise ValueError('A conflicted initializer must be an actual registered original camera')
    coarse_initializers = additions | missing_initial | unsupported_original | conflicted
    if not coarse_initializers <= coarse_names:
        raise ValueError('An unobserved required frame or bridge lacks a qualified raw coarse pose')
    return {'required_names': sorted(original | initial_names),
        'original_registered_names': sorted(original),
        'missing_initial_names': sorted(missing_initial), 'optional_bridge_names': sorted(additions),
        'visual_initializer_names': sorted(original - unsupported_original - conflicted),
        'coarse_initializer_names': sorted(coarse_initializers),
        'unobserved_original_names': sorted(unobserved_original),
        'unsupported_original_names': sorted(unsupported_original),
        'degenerate_observed_original_names': sorted(unsupported_original - unobserved_original),
        'unverified_conflicted_original_names': sorted(conflicted)}


def observation_decisions(observations, required_names, optional_names):
    required, optional = set(required_names), set(optional_names)
    if required & optional or set(observations) - required - optional:
        raise ValueError('Camera observation decisions have an inconsistent role partition')
    rejected = sorted(name for name in optional if observations.get(name, 0) == 0)
    unsupported = sorted(name for name in required if observations.get(name, 0) == 0)
    return {'rejected_optional_names': rejected, 'unsupported_required_names': unsupported,
        'zero_observation_guard_unchanged': True, 'required_frames_may_be_silently_dropped': False}


def point_support_roles(registered_names, initial_names, seeded=None, *, allow_pending_point_rebuild=False):
    """Keep optional witnesses distinct from required cameras after point filtering."""
    registered, initial = set(registered_names), set(initial_names)
    if seeded is None:
        return {'required_names': sorted(registered), 'optional_names': [],
            'scope': 'all_registered_cameras_required_default_strict'}
    if seeded.get('status') == NATIVE_POINT_REBUILD_INPUT_STATUS:
        if (not allow_pending_point_rebuild or seeded.get('full_native_point_support_pending') is not True
                or seeded.get('training_admission_granted') is not False):
            raise ValueError('Provisional registration requires actual all-point rebuilding before BA support')
    elif seeded.get('status') not in VISUAL_SUPPORT_STATUSES:
        raise ValueError('Typed point support requires actual visual support accounting')
    roles = seeded['initialization_roles']
    original, required, optional = map(set, (roles['original_registered_names'],
        roles['required_names'], roles['optional_bridge_names']))
    if (required != original | initial or required & optional or not required <= registered
            or registered - required - optional or seeded.get('required_initial_names_all_registered') is not True):
        raise ValueError('Point support roles changed a required initial or original camera denominator')
    return {'required_names': sorted(required), 'optional_names': sorted(optional),
        'scope': 'same_attempt_required_cameras_optional_bridge_witnesses'}

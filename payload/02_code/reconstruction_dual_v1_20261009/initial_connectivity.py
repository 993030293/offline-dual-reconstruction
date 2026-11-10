"""Check whether nominated raw views belong to one genuine matching component."""

PAIR_BASE = 2147483647


def matching_components(connection, names, minimum_pair_inliers):
    """Ignore unnominated views and weak edges when assessing scene connectivity."""
    names = set(names)
    images = {int(key): name for key, name in connection.execute('SELECT image_id,name FROM images')
        if name in names}
    if not names or set(images.values()) != names or minimum_pair_inliers < 1:
        raise ValueError('Connectivity requires actual nominated images and a positive native threshold')
    parents = {key: key for key in images}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for pair, inliers in connection.execute('SELECT pair_id,rows FROM two_view_geometries WHERE rows>=?',
            (minimum_pair_inliers,)):
        first, second = divmod(int(pair), PAIR_BASE)
        if first in images and second in images:
            parents[find(first)] = find(second)
    groups = {}
    for key, name in images.items():
        groups.setdefault(find(key), []).append(name)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: (-len(group), group))


def component_recovery_pairs(connection, components):
    """Attempt every missing inter-component pair before rejecting an isolated group."""
    names = [name for group in components for name in group]
    if not components or any(not group for group in components) or len(set(names)) != len(names):
        raise ValueError('Connectivity recovery requires a disjoint nonempty component partition')
    nominated = set(names)
    images = {name: int(key) for key, name in connection.execute('SELECT image_id,name FROM images')
        if name in nominated}
    if set(images) != nominated:
        raise ValueError('A connectivity recovery image has no actual database feature identity')
    membership = {images[name]: index for index, group in enumerate(components) for name in group}
    by_id = {key: name for name, key in images.items()}
    attempted = {int(pair) for pair, in connection.execute('SELECT pair_id FROM matches')}
    keys = sorted(membership)
    return [(by_id[first], by_id[second]) for index, first in enumerate(keys) for second in keys[index + 1:]
        if membership[first] != membership[second] and PAIR_BASE * first + second not in attempted]


def main_component(components):
    """Abstain when the dominant scene cannot be determined from component size."""
    if not components or any(not group for group in components):
        raise ValueError('No qualified matching component is available')
    ordered = sorted((set(group) for group in components), key=lambda group: (-len(group), sorted(group)))
    if sum(map(len, ordered)) != len(set.union(*ordered)):
        raise ValueError('Matching component identities overlap')
    if len(ordered) > 1 and len(ordered[0]) == len(ordered[1]):
        raise ValueError('Ambiguous equally sized disconnected visual scenes require abstention')
    return ordered[0]

"""Validate live birth identities along every observed process ancestry edge."""

import psutil


def capture_owned_tree(root_pid, root_birth):
    """Reject stale PPID links before admitting or later terminating a process."""
    root = psutil.Process(root_pid)
    if root.create_time() != root_birth:
        raise RuntimeError('The owned root process identity changed during monitoring')
    accepted = [root]
    rejected = []
    for candidate in root.children(recursive=True):
        try:
            candidate_birth = candidate.create_time()
            node, seen = candidate, set()
            reason = None
            while (node.pid, node.create_time()) != (root_pid, root_birth):
                node_birth = node.create_time()
                identity = (node.pid, node_birth)
                if identity in seen:
                    raise RuntimeError('An observed process ancestry contains a cycle')
                seen.add(identity)
                parent_pid = node.ppid()
                parent = psutil.Process(parent_pid)
                parent_birth = parent.create_time()
                if node_birth < parent_birth:
                    reason = {'pid': candidate.pid, 'creation_time': candidate_birth,
                        'reason': 'child_predates_reused_parent', 'edge_child_pid': node.pid,
                        'edge_child_birth': node_birth, 'parent_pid': parent_pid,
                        'parent_birth': parent_birth}
                    break
                node = parent
            if reason is not None:
                rejected.append(reason)
            elif candidate.create_time() == candidate_birth:
                accepted.append(candidate)
        except psutil.NoSuchProcess:
            # An exiting descendant cannot establish a new live ownership identity.
            continue
    if root.create_time() != root_birth:
        raise RuntimeError('The owned root changed during ancestry validation')
    return accepted, rejected

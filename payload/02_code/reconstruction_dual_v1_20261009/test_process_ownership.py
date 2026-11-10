"""Exercise stale parent PID reuse without touching operating-system processes."""

import unittest
from unittest.mock import patch

from common import OLD
import process_ownership as ownership


class Process:
    def __init__(self, pid, birth, parent):
        self.pid, self.birth, self.parent_pid = pid, birth, parent
        self.descendants = []

    def create_time(self):
        return self.birth

    def ppid(self):
        return self.parent_pid

    def children(self, recursive=False):
        return self.descendants


class ProcessOwnershipTests(unittest.TestCase):
    def collect(self, nodes, descendants, root_birth=10):
        nodes[1].descendants = [nodes[pid] for pid in descendants]
        with patch.object(ownership.psutil, 'Process', side_effect=lambda pid: nodes[pid]):
            return ownership.capture_owned_tree(1, root_birth)

    def test_valid_nested_children_are_preserved(self):
        nodes = {1: Process(1, 10, 0), 2: Process(2, 11, 1), 3: Process(3, 12, 2)}
        owned, rejected = self.collect(nodes, [2, 3])
        self.assertEqual([p.pid for p in owned], [1, 2, 3])
        self.assertEqual(rejected, [])

    def test_old_orphan_of_reused_intermediate_pid_is_not_owned(self):
        nodes = {1: Process(1, 10, 0), 2: Process(2, 30, 1), 3: Process(3, 20, 2)}
        owned, rejected = self.collect(nodes, [2, 3])
        self.assertEqual([p.pid for p in owned], [1, 2])
        self.assertEqual(rejected[0]['pid'], 3)
        self.assertEqual(rejected[0]['reason'], 'child_predates_reused_parent')

    def test_temporal_check_covers_every_ancestor_edge(self):
        nodes = {1: Process(1, 10, 0), 2: Process(2, 30, 1),
                 3: Process(3, 20, 2), 4: Process(4, 40, 3)}
        owned, rejected = self.collect(nodes, [2, 3, 4])
        self.assertEqual([p.pid for p in owned], [1, 2])
        self.assertEqual({row['pid'] for row in rejected}, {3, 4})

    def test_equal_kernel_birth_times_are_allowed(self):
        nodes = {1: Process(1, 10, 0), 2: Process(2, 10, 1)}
        owned, rejected = self.collect(nodes, [2])
        self.assertEqual([p.pid for p in owned], [1, 2])
        self.assertEqual(rejected, [])

    def test_reused_root_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'root process identity'):
            self.collect({1: Process(1, 20, 0)}, [], root_birth=10)

    def test_cyclic_ancestry_is_rejected(self):
        nodes = {1: Process(1, 10, 0), 2: Process(2, 20, 3), 3: Process(3, 20, 2)}
        with self.assertRaisesRegex(RuntimeError, 'ancestry contains a cycle'):
            self.collect(nodes, [2])


if __name__ == '__main__':
    unittest.main()

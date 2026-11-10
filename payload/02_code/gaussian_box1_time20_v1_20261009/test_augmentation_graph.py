"""Check image preservation, edge ownership, deterministic ties and loop coverage."""

import unittest
from types import SimpleNamespace as NS

from augmentation_graph import proposal_pairs


class GraphContractTest(unittest.TestCase):
    def fixture(self):
        names = [f"old_{i:03d}" for i in range(100)] + ["new_a", "new_b"]
        rows = {name: {"camera_header_ns": i*100000000} for i, name in enumerate(names[:100])}
        rows.update(new_a={"camera_header_ns": 4500000000}, new_b={"camera_header_ns": 4600000000})
        images = {i: NS(image_id=i, name=names[i], has_pose=True, points2D=[]) for i in range(100)}
        point = NS(point3D_id=1, has_point3D=lambda: True)
        images[45].points2D = [point]
        track = NS(elements=[NS(image_id=45), NS(image_id=0)])
        previous = NS(images=images, points3D={1: NS(track=track)})
        return names, rows, previous

    def test_complete_names_and_only_new_owned_edges(self):
        names, rows, previous = self.fixture()
        original = list(names)
        new = {"new_a": rows["new_a"], "new_b": rows["new_b"]}
        pairs = proposal_pairs(names, new, rows, previous)
        self.assertEqual(names, original)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(a < b and ({a, b} & set(new)) for a, b in pairs))
        self.assertIn(("new_a", "new_b"), pairs)
        self.assertEqual({n for p in pairs for n in p if n in new}, set(new))

    def test_covisible_loop_outside_temporal_window_is_retained(self):
        names, rows, previous = self.fixture()
        pairs = proposal_pairs(names, {"new_a": rows["new_a"]}, rows, previous)
        self.assertIn(tuple(sorted(("old_000", "new_a"))), pairs)
        self.assertNotIn(tuple(sorted(("old_099", "new_a"))), pairs)

    def test_order_invariance_and_bad_identity_rejection(self):
        names, rows, previous = self.fixture()
        new = {"new_a": rows["new_a"], "new_b": rows["new_b"]}
        self.assertEqual(proposal_pairs(names, new, rows, previous),
                         proposal_pairs(list(reversed(names)), dict(reversed(list(new.items()))), rows, previous))
        with self.assertRaises(ValueError):
            proposal_pairs(names+[names[0]], new, rows, previous)


if __name__ == "__main__":
    unittest.main()

"""Reject shifted payloads and preserve binary PCD data boundaries."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from public_gt_audit import inspect_tls


class BinaryPCDLayout(unittest.TestCase):
    def test_binary_starts_after_data_and_zero_tail_is_not_header_padding(self):
        header=b"VERSION 0.7\nFIELDS x y z _\nSIZE 4 4 4 1\nTYPE F F F U\nCOUNT 1 1 1 4\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n"
        points=np.array([(1.,2.,3.,0),(4.,5.,6.,0)],dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("padding","<u4")])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"sample.pcd"
            path.write_bytes(header+points.tobytes()+bytes(100))
            receipt,sample=inspect_tls(path)
            self.assertEqual(receipt["payload_offset"],len(header))
            self.assertEqual(receipt["verified_zero_trailing_bytes"],100)
            np.testing.assert_allclose(sample,[[1,2,3],[4,5,6]])
            path.write_bytes(header+points.tobytes()+b"invalid")
            with self.assertRaises(ValueError): inspect_tls(path)


if __name__ == "__main__":
    unittest.main()

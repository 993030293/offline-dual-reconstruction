import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "gaussian_full_raw_ab_20261008" / "pack_export.js"


class PackExportReceiptTests(unittest.TestCase):
    def test_acknowledgement_body_is_consumed_and_validated_before_completion(self):
        code = SOURCE.read_text(encoding="utf-8")
        body = 'const receiptBody = await receipt.text();'
        acknowledgement = 'receiptResult.status !== "native_array_and_complete_asset_oracles_recorded"'
        completion = 'status.dataset.complete = "true";'
        self.assertEqual(code.count(body), 1)
        self.assertEqual(code.count(acknowledgement), 1)
        self.assertEqual(code.count(completion), 1)
        self.assertLess(code.index(body), code.index(acknowledgement))
        self.assertLess(code.index(acknowledgement), code.index(completion))


if __name__ == "__main__":
    unittest.main()

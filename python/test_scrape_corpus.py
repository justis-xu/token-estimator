import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(__file__))

sys.modules["requests"] = types.SimpleNamespace()

import scrape_corpus


class ManualSamplesTest(unittest.TestCase):
    def test_includes_whitespace_heavy_calibration_samples(self):
        texts = "\n".join(sample["text"] for sample in scrape_corpus.MANUAL_SAMPLES)

        self.assertIn('"messages": [', texts)
        self.assertIn("```go", texts)
        self.assertIn("| 指标 | 当前值 |", texts)


if __name__ == "__main__":
    unittest.main()

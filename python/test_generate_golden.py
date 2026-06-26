import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(__file__))

sys.modules["calculate_discount"] = types.SimpleNamespace(
    real_count=lambda model_key, text: len(text)
)

import generate_golden


class GenerateGoldenRateLimitTest(unittest.TestCase):
    def test_sleeps_after_each_api_model_count_only(self):
        corpus = [{"text": "一"}, {"text": "二"}]
        calls = []
        sleeps = []

        def fake_real_count(model_key, text):
            calls.append((model_key, text))
            return len(text)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "golden.jsonl")
            total = generate_golden.write_golden(
                corpus,
                ["doubao", "gpt-4"],
                out_path,
                real_count_fn=fake_real_count,
                sleep_fn=sleeps.append,
            )

        self.assertEqual(total, 4)
        self.assertEqual(calls, [
            ("doubao", "一"),
            ("doubao", "二"),
            ("gpt-4", "一"),
            ("gpt-4", "二"),
        ])
        self.assertEqual(sleeps, [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()

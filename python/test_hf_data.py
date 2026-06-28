import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import config
import hf_data


class HFDataTest(unittest.TestCase):
    def test_prefers_existing_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_output = config.OUTPUT_DIR
            old_repo = config.HF_DATASET_REPO
            config.OUTPUT_DIR = tmp
            hf_data.OUTPUT_DIR = tmp
            config.HF_DATASET_REPO = ""
            hf_data.HF_DATASET_REPO = ""

            path = os.path.join(tmp, "corpus.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("ok")

            self.assertEqual(hf_data.ensure_output_file("corpus.jsonl"), path)

            config.OUTPUT_DIR = old_output
            hf_data.OUTPUT_DIR = old_output
            config.HF_DATASET_REPO = old_repo
            hf_data.HF_DATASET_REPO = old_repo

    def test_downloads_from_hf_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_output = config.OUTPUT_DIR
            old_repo = config.HF_DATASET_REPO
            old_revision = config.HF_DATASET_REVISION
            old_subdir = config.HF_DATASET_SUBDIR
            config.OUTPUT_DIR = tmp
            hf_data.OUTPUT_DIR = tmp
            config.HF_DATASET_REPO = "owner/repo"
            hf_data.HF_DATASET_REPO = "owner/repo"
            config.HF_DATASET_REVISION = "main"
            hf_data.HF_DATASET_REVISION = "main"
            config.HF_DATASET_SUBDIR = "calibration"
            hf_data.HF_DATASET_SUBDIR = "calibration"

            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = os.path.join(tmp, "corpus.jsonl")
                with open(target, "w", encoding="utf-8") as f:
                    f.write("downloaded")
                return target

            sys.modules["huggingface_hub"] = types.SimpleNamespace(hf_hub_download=fake_download)
            path = hf_data.ensure_output_file("corpus.jsonl")

            self.assertEqual(path, os.path.join(tmp, "corpus.jsonl"))
            self.assertEqual(calls[0]["repo_id"], "owner/repo")
            self.assertEqual(calls[0]["repo_type"], "dataset")
            self.assertEqual(calls[0]["filename"], "calibration/corpus.jsonl")

            del sys.modules["huggingface_hub"]
            config.OUTPUT_DIR = old_output
            hf_data.OUTPUT_DIR = old_output
            config.HF_DATASET_REPO = old_repo
            hf_data.HF_DATASET_REPO = old_repo
            config.HF_DATASET_REVISION = old_revision
            hf_data.HF_DATASET_REVISION = old_revision
            config.HF_DATASET_SUBDIR = old_subdir
            hf_data.HF_DATASET_SUBDIR = old_subdir


if __name__ == "__main__":
    unittest.main()

"""
Upload/download calibration artifacts to/from a Hugging Face dataset repo.

Examples:
  export HF_DATASET_REPO=justis-xu/token-estimator-data
  export HF_TOKEN=hf_xxx
  python sync_hf_dataset.py upload corpus.jsonl golden.jsonl
  python sync_hf_dataset.py download corpus.jsonl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import OUTPUT_DIR, HF_DATASET_REPO, HF_DATASET_REVISION, HF_DATASET_SUBDIR, HF_TOKEN
from hf_data import ensure_output_file, local_output_path


def _remote_path(filename: str) -> str:
    return f"{HF_DATASET_SUBDIR.rstrip('/')}/{filename}" if HF_DATASET_SUBDIR else filename


def upload(files: list[str]) -> None:
    if not HF_DATASET_REPO:
        raise EnvironmentError("HF_DATASET_REPO is not set")
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN is not set")

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)

    for filename in files:
        local_path = local_output_path(filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"{local_path} not found")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=_remote_path(filename),
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )
        print(f"uploaded: {local_path} -> {HF_DATASET_REPO}/{_remote_path(filename)}")


def download(files: list[str]) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for filename in files:
        path = ensure_output_file(filename)
        print(f"downloaded: {filename} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync output artifacts with a Hugging Face dataset repo")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("files", nargs="*", default=["corpus.jsonl", "golden.jsonl"])
    args = parser.parse_args()

    print(f"repo={HF_DATASET_REPO or '(unset)'} revision={HF_DATASET_REVISION}")
    if args.action == "upload":
        upload(args.files)
    else:
        download(args.files)


if __name__ == "__main__":
    main()

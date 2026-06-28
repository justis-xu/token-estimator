"""
Helpers for syncing calibration artifacts with a Hugging Face dataset repo.

Behavior is intentionally simple:
1. Prefer local files in OUTPUT_DIR.
2. If missing and HF_DATASET_REPO is set, download from the dataset repo.
3. Upload explicitly via sync_hf_dataset.py.
"""

from __future__ import annotations

import os
from pathlib import Path

from config import OUTPUT_DIR, HF_DATASET_REPO, HF_DATASET_REVISION, HF_DATASET_SUBDIR


def _remote_path(filename: str) -> str:
    return f"{HF_DATASET_SUBDIR.rstrip('/')}/{filename}" if HF_DATASET_SUBDIR else filename


def local_output_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


def ensure_output_file(filename: str) -> str:
    path = local_output_path(filename)
    if os.path.exists(path):
        return path
    if not HF_DATASET_REPO:
        raise FileNotFoundError(
            f"{path} not found, and HF_DATASET_REPO is not set for fallback download"
        )

    # Prefer the plain Hub download path over Xet on local developer machines.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import hf_hub_download

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=_remote_path(filename),
        revision=HF_DATASET_REVISION,
        local_dir=OUTPUT_DIR,
        token=os.environ.get("HF_TOKEN") or None,
    )
    local_path = Path(downloaded)
    if local_path.name == filename:
        return str(local_path)

    expected = Path(path)
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(local_path.read_bytes())
    return str(expected)

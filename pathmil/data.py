"""Disk-based datasets and sample-info helpers (ported from train_mil_loo.py).

Combined per-sample H5 files store:
  - 'embeddings'      (n_spots, n_instances, input_dim)
  - 'gene_expression' (n_spots, n_outputs)   [target; genes OR module scores]
  - attrs['n_spots']
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MILRegressionDiskDataset(Dataset):
    """Concatenation of per-sample combined H5 files, loaded on the fly."""

    def __init__(self, sample_info_list: list):
        self.samples = []
        self.lib_ids = []
        self.cumulative_indices = [0]

        for info in sample_info_list:
            if info is not None:
                self.samples.append(info)
                self.lib_ids.extend([info["library_id"]] * info["n_spots"])
                self.cumulative_indices.append(
                    self.cumulative_indices[-1] + info["n_spots"])

        self.lib_ids = np.array(self.lib_ids)
        self.total_spots = self.cumulative_indices[-1]

    def __len__(self) -> int:
        return self.total_spots

    def __getitem__(self, idx: int):
        sample_idx = np.searchsorted(self.cumulative_indices[1:], idx, side="right")
        local_idx = idx - self.cumulative_indices[sample_idx]

        file_path = self.samples[sample_idx]["file_path"]
        with h5py.File(file_path, "r") as hf:
            X = hf["embeddings"][local_idx]
            y = hf["gene_expression"][local_idx]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        lib_id = self.samples[sample_idx]["library_id"]
        return X_tensor, y_tensor, lib_id

    def get_sample_indices(self, lib_id: str) -> np.ndarray:
        return np.where(self.lib_ids == lib_id)[0]

    def target_mean(self, max_spots: int = 2000) -> float:
        """Mean target value over a sample of spots, for output bias init.

        Build this from the TRAINING samples only, or the held-out sample's mean
        leaks into where the model starts.
        """
        step = max(1, self.total_spots // max_spots)
        vals = [float(self[i][1].mean()) for i in range(0, self.total_spots, step)]
        return float(np.mean(vals)) if vals else 0.0


class MILTestDataset(Dataset):
    """Embeddings-only dataset for prediction on new samples (no targets)."""

    def __init__(self, sample_info: dict):
        self.library_id = sample_info["library_id"]
        self.file_path = sample_info["file_path"]
        self.n_spots = sample_info["n_spots"]

    def __len__(self) -> int:
        return self.n_spots

    def __getitem__(self, idx: int):
        with h5py.File(self.file_path, "r") as hf:
            X = hf["embeddings"][idx]
        return torch.tensor(X, dtype=torch.float32), self.library_id


def load_sample_info_from_directory(processed_dir, library_ids=None) -> list:
    """List combined H5 files in `processed_dir` as sample-info dicts.

    Args:
        processed_dir: dir containing <library_id>.h5 combined files.
        library_ids:   optional subset to keep.
    """
    processed_dir = Path(processed_dir)
    sample_info = []
    for h5_file in sorted(processed_dir.glob("*.h5")):
        lib_id = h5_file.stem
        if library_ids is not None and lib_id not in library_ids:
            continue
        with h5py.File(h5_file, "r") as hf:
            n_spots = int(hf.attrs["n_spots"])
        sample_info.append({
            "library_id": lib_id,
            "n_spots": n_spots,
            "file_path": str(h5_file),
        })
    return sample_info


def embedding_sample_info(embeddings_dir, library_id,
                          suffix="_patch_embeddings.h5") -> dict:
    """Sample-info dict for a raw `<lib><suffix>` embeddings file (predict path).

    `suffix` selects the track: "_patch_embeddings.h5" (spot bags, default) or
    "_image_patch_embeddings.h5" (no-gap sliding-window grid for visualization).
    """
    h5_file = Path(embeddings_dir) / f"{library_id}{suffix}"
    if not h5_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {h5_file}")
    with h5py.File(h5_file, "r") as hf:
        n_spots = int(hf["embeddings"].shape[0])
    return {"library_id": library_id, "n_spots": n_spots, "file_path": str(h5_file)}

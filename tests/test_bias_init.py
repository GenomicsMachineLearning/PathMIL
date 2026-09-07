"""Seam: where a non-negative head starts.

`softplus(0) = 0.693` against a log1p-normalised target mean of ~0.016 is a ~43x
overshoot, repeated across every output column, so the first epochs are spent
walking the bias down rather than learning. Initialising the output bias to the
training-target mean is what makes `output_activation=softplus` usable.

The bias must come from the TRAINING samples only. Deriving it from the whole
dataset would leak the held-out sample's target mean into the model's starting
point -- small, but it is leakage in a number we then report.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from pathmil.data import MILRegressionDiskDataset
from pathmil.train import _resolve_bias_init


def _sample(tmp_path, lib_id, values):
    """A processed H5 whose gene_expression is exactly `values`."""
    values = np.asarray(values, dtype=np.float32)
    n_spots, n_genes = values.shape
    path = tmp_path / f"{lib_id}.h5"
    with h5py.File(path, "w") as hf:
        hf.create_dataset("embeddings", data=np.zeros((n_spots, 2, 4), dtype=np.float32))
        hf.create_dataset("gene_expression", data=values)
        hf.attrs["n_spots"] = n_spots
        hf.attrs["library_id"] = lib_id
    return {"library_id": lib_id, "file_path": str(path), "n_spots": n_spots}


class TestTargetMean:
    def test_target_mean_is_the_mean_over_every_target_and_spot(self, tmp_path):
        # Spot means are 2.0 and 4.0, so the dataset mean is 3.0.
        info = [_sample(tmp_path, "lib1", [[1.0, 3.0], [3.0, 5.0]])]
        ds = MILRegressionDiskDataset(info)
        assert ds.target_mean() == pytest.approx(3.0)

    def test_target_mean_spans_all_samples(self, tmp_path):
        info = [_sample(tmp_path, "lib1", [[0.0, 0.0]]),
                _sample(tmp_path, "lib2", [[4.0, 4.0]])]
        ds = MILRegressionDiskDataset(info)
        assert ds.target_mean() == pytest.approx(2.0)

    def test_empty_dataset_returns_zero_rather_than_nan(self, tmp_path):
        ds = MILRegressionDiskDataset([])
        assert ds.target_mean() == 0.0


class TestResolveBiasInit:
    def test_derives_the_bias_from_the_training_target_mean(self, tmp_path):
        ds = MILRegressionDiskDataset([_sample(tmp_path, "lib1", [[1.0, 3.0]])])
        cfg = {"model": {"bias_init_from_data": True}}
        assert _resolve_bias_init(cfg, ds) == pytest.approx(2.0)

    def test_derivation_is_the_default(self, tmp_path):
        ds = MILRegressionDiskDataset([_sample(tmp_path, "lib1", [[1.0, 3.0]])])
        assert _resolve_bias_init({}, ds) == pytest.approx(2.0)

    def test_disabling_it_falls_back_to_the_explicit_config_value(self, tmp_path):
        ds = MILRegressionDiskDataset([_sample(tmp_path, "lib1", [[1.0, 3.0]])])
        cfg = {"model": {"bias_init_from_data": False, "output_bias_init": 0.5}}
        assert _resolve_bias_init(cfg, ds) == pytest.approx(0.5)

    def test_disabling_it_with_no_explicit_value_leaves_the_torch_default(self, tmp_path):
        ds = MILRegressionDiskDataset([_sample(tmp_path, "lib1", [[1.0, 3.0]])])
        cfg = {"model": {"bias_init_from_data": False}}
        assert _resolve_bias_init(cfg, ds) is None

    def test_derived_value_wins_over_an_explicit_one(self, tmp_path):
        # The derived value overwrites output_bias_init. Two sources of truth
        # for one number would be worse.
        ds = MILRegressionDiskDataset([_sample(tmp_path, "lib1", [[1.0, 3.0]])])
        cfg = {"model": {"bias_init_from_data": True, "output_bias_init": 99.0}}
        assert _resolve_bias_init(cfg, ds) == pytest.approx(2.0)

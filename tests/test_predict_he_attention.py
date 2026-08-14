"""Seam: the attention axis in `scripts/predict_he.py --sc_pred`.

Under shared attention a nucleus has one attention weight; under per-gene
attention it has one *per target*. Both the narrowing and the per-cell table have
to follow that, and neither failure mode raises: collapsing the target axis with
a bare `.mean()` returns a plausible-looking scalar, and forgetting to narrow the
attention array leaves the dominant memory term outside the `--max-instance-gb`
estimate. These tests pin both.

`--no-save-attention` is the other half: under a per-gene checkpoint the attention
array is the same size as the instance store, so dropping it has to drop the whole
term -- the accumulation, the file, and the cell-table columns -- not just the
`np.save`. Skipping only the write would leave peak memory unchanged, which is the
thing the flag exists to fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import predict_he  # noqa: E402


class TestAttentionNarrowing:
    """`--targets` must narrow the attention array by the same columns as the scores."""

    @staticmethod
    def _narrow(attn, inst_cols):
        # The transform applied in predict_slide.flush().
        return attn[:, :, inst_cols] if attn.ndim == 3 else attn

    def test_per_gene_attention_is_narrowed_to_the_scored_targets(self):
        attn = np.random.rand(2, 256, 50).astype("float32")
        got = self._narrow(attn, [3, 7])
        assert got.shape == (2, 256, 2)
        np.testing.assert_array_equal(got[:, :, 0], attn[:, :, 3])
        np.testing.assert_array_equal(got[:, :, 1], attn[:, :, 7])

    def test_shared_attention_is_left_alone(self):
        attn = np.random.rand(2, 256).astype("float32")
        np.testing.assert_array_equal(self._narrow(attn, [3, 7]), attn)

    def test_narrowed_attention_matches_the_instance_array_shape(self):
        # They are written as a matched pair; a shape mismatch means one of them
        # describes a different target set from the other.
        inst = np.random.rand(2, 256, 50).astype("float32")[:, :, [3, 7]]
        assert self._narrow(np.random.rand(2, 256, 50).astype("float32"),
                            [3, 7]).shape == inst.shape


class TestPerCellAttention:
    """A nucleus's attention, aggregated over the tokens its outline covers."""

    @staticmethod
    def _cell_attn(attn, toks):
        # The expression used in _score_cells_in_tile.
        return attn[toks].mean(axis=0) if attn.ndim == 2 else float(attn[toks].mean())

    def test_shared_attention_gives_one_weight_per_nucleus(self):
        attn = np.arange(256, dtype="float32")
        got = self._cell_attn(attn, np.array([0, 1, 2]))
        assert isinstance(got, float) and got == pytest.approx(1.0)

    def test_per_gene_attention_gives_one_weight_per_target(self):
        attn = np.zeros((256, 3), dtype="float32")
        attn[:, 0], attn[:, 1], attn[:, 2] = 1.0, 2.0, 3.0
        got = self._cell_attn(attn, np.array([5, 9]))
        assert got.shape == (3,)
        np.testing.assert_allclose(got, [1.0, 2.0, 3.0])

    def test_targets_are_not_averaged_together(self):
        # The regression: `.mean()` with no axis returns 2.0 here -- one number
        # that belongs to no target in particular.
        attn = np.zeros((256, 3), dtype="float32")
        attn[:, 0], attn[:, 1], attn[:, 2] = 1.0, 2.0, 3.0
        got = self._cell_attn(attn, np.array([5, 9]))
        assert np.ndim(got) == 1, "target axis was collapsed"


class TestCellTable:
    @staticmethod
    def _sc(attention, n_cells=4, n_targets=3):
        return {
            "cell_xy": np.zeros((n_cells, 2)),
            "cell_area": np.ones(n_cells),
            "cell_score": np.zeros((n_cells, n_targets)),
            "cell_attention": attention,
            "cell_poly": [np.zeros((4, 2))] * n_cells,
        }

    def test_shared_attention_writes_one_attention_column(self, tmp_path):
        names = ["A", "B", "C"]
        predict_he.write_cell_table("s", names, self._sc(np.ones(4)), tmp_path)
        cols = list(pd.read_csv(tmp_path / "cells.csv").columns)
        assert cols == ["cell_id", "x", "y", "area", "attention", "A", "B", "C"]

    def test_per_gene_attention_writes_one_column_per_target(self, tmp_path):
        names = ["A", "B", "C"]
        attn = np.tile(np.array([0.1, 0.2, 0.3]), (4, 1))
        predict_he.write_cell_table("s", names, self._sc(attn), tmp_path)
        df = pd.read_csv(tmp_path / "cells.csv")
        assert list(df.columns) == ["cell_id", "x", "y", "area", "A", "B", "C",
                                    "attention_A", "attention_B", "attention_C"]
        np.testing.assert_allclose(df["attention_B"], 0.2)

    def test_identifier_columns_keep_their_positions(self, tmp_path):
        # Consumers index the leading columns positionally; the per-target block
        # is appended so it cannot shift them.
        attn = np.tile(np.array([0.1, 0.2, 0.3]), (4, 1))
        predict_he.write_cell_table("s", ["A", "B", "C"], self._sc(attn), tmp_path)
        df = pd.read_csv(tmp_path / "cells.csv")
        assert list(df.columns)[:4] == ["cell_id", "x", "y", "area"]

    def test_no_attention_columns_when_attention_was_dropped(self, tmp_path):
        # --no-save-attention: `cell_attention` is None rather than an empty array,
        # so the table has to omit the block entirely instead of writing NaNs.
        predict_he.write_cell_table("s", ["A", "B", "C"], self._sc(None), tmp_path)
        cols = list(pd.read_csv(tmp_path / "cells.csv").columns)
        assert cols == ["cell_id", "x", "y", "area", "A", "B", "C"]
        assert not [c for c in cols if c.startswith("attention")]


class TestInstanceStoreEstimate:
    """`--max-instance-gb` must count exactly the arrays that are actually held."""

    ONE = 1000 * 256 * 50 * 4 / 1e9   # one (n_tiles, 256, n_targets) float32 array

    def test_shared_attention_counts_the_instance_array_only(self):
        got = predict_he._instance_store_gb(1000, 50, per_gene=False, keep_attention=True)
        assert got == pytest.approx(self.ONE)

    def test_per_gene_attention_counts_a_second_array(self):
        got = predict_he._instance_store_gb(1000, 50, per_gene=True, keep_attention=True)
        assert got == pytest.approx(2 * self.ONE)

    def test_dropping_attention_halves_the_per_gene_estimate(self):
        got = predict_he._instance_store_gb(1000, 50, per_gene=True, keep_attention=False)
        assert got == pytest.approx(self.ONE)

    def test_dropping_attention_does_not_change_the_shared_estimate(self):
        # Shared attention is (n_tiles, 256) -- never part of the estimate, so the
        # flag must not appear to buy anything here.
        assert predict_he._instance_store_gb(1000, 50, per_gene=False,
                                             keep_attention=False) == pytest.approx(self.ONE)


class TestNoSaveAttentionFlag:
    @staticmethod
    def _parse(*extra):
        return predict_he.build_parser().parse_args(
            ["--slide", "s.svs", "--checkpoint", "c.pth", "--out", "o", *extra])

    def test_attention_is_saved_by_default(self):
        assert self._parse().no_save_attention is False

    def test_flag_turns_it_off(self):
        assert self._parse("--no-save-attention").no_save_attention is True


class TestScoreCellsWithoutAttention:
    """`_score_cells_in_tile` still scores nuclei when attention is not collected."""

    @staticmethod
    def _args():
        return argparse.Namespace(seg_prob_thresh=0.2, seg_nms_thresh=0.6,
                                  min_cell_area=0.0)

    @staticmethod
    def _patch_segmentation(monkeypatch):
        # One nucleus at the tile centre, as a 4-point square outline.
        def fake(seg_model, crop, prob_thresh, nms_thresh):
            return (np.array([[112.0, 112.0]]),
                    [np.array([[108., 108.], [116., 108.], [116., 116.], [108., 116.]])],
                    np.array([64.0]))
        monkeypatch.setattr(predict_he, "_segment_crop", fake)

    def _run(self, monkeypatch, attn):
        self._patch_segmentation(monkeypatch)
        inst = np.arange(256 * 2, dtype="float32").reshape(256, 2)
        cell_xy, cell_area, cell_poly, cell_score, cell_attn = [], [], [], [], []
        predict_he._score_cells_in_tile(
            object(), None, 0, 0, 0, 0, 224, 224 / 16, inst, attn, self._args(),
            cell_xy, cell_area, cell_poly, cell_score, cell_attn)
        return cell_score, cell_attn

    def test_scores_are_still_collected(self, monkeypatch):
        cell_score, _ = self._run(monkeypatch, None)
        assert len(cell_score) == 1 and cell_score[0].shape == (2,)

    def test_no_attention_is_collected(self, monkeypatch):
        _, cell_attn = self._run(monkeypatch, None)
        assert cell_attn == []

    def test_attention_is_collected_when_given(self, monkeypatch):
        _, cell_attn = self._run(monkeypatch, np.ones((256, 2), dtype="float32"))
        assert len(cell_attn) == 1


class _FakeSlide:
    """Two 224px tiles side by side, dark enough to pass the tissue filter."""

    width, height, mpp = 448, 224, None

    def read_region(self, x, y, size):
        from PIL import Image
        return Image.new("RGB", (int(size), int(size)), (30, 30, 30))


class _FakeMIL:
    """Per-gene attention head over 3 outputs; values encode their target index."""

    attention_mode = "per_gene"

    def __call__(self, tokens, return_instance_predictions=False):
        import torch
        b, n = tokens.shape[0], tokens.shape[1]
        inst = torch.arange(3, dtype=torch.float32).expand(b, n, 3).contiguous()
        attn = torch.full((b, n, 3), 1.0 / n)
        bag = (inst * attn).sum(dim=1)
        if return_instance_predictions:
            return bag, inst, attn
        return bag


class TestPredictSlideWithoutAttention:
    """End to end through `predict_slide`: the flag must stop the accumulation.

    Skipping only the `np.save` would leave `attn_all` growing tile by tile, so peak
    memory -- the reason the flag exists under a per-gene checkpoint -- would be
    unchanged. Asserting on the returned `sc` is what catches that.
    """

    @staticmethod
    def _args(no_save_attention):
        return argparse.Namespace(
            tile_size=224, stride=224, target_mpp=None, batch_size=2, min_tissue=0.1,
            white_threshold=220, max_tiles=None, seg_margin=0, min_cell_area=0.0,
            dedup_radius=0.0, seg_prob_thresh=0.2, seg_nms_thresh=0.6,
            max_instance_gb=8.0, no_save_attention=no_save_attention)

    def _run(self, monkeypatch, no_save_attention):
        import torch

        # No nuclei: this test is about the per-tile stores, not the cell table.
        monkeypatch.setattr(predict_he, "_segment_crop",
                            lambda *a, **k: (np.zeros((0, 2)), [], np.zeros(0)))
        virchow = lambda x: torch.zeros(x.shape[0], 5 + 256, 8)   # noqa: E731
        transform = lambda img: torch.zeros(3, 224, 224)          # noqa: E731
        return predict_he.predict_slide(
            Path("fake.svs"), _FakeSlide(), virchow, transform, _FakeMIL(),
            self._args(no_save_attention), torch.device("cpu"), inst_cols=[0, 2],
            seg_model=object())

    def test_attention_is_returned_by_default(self, monkeypatch):
        _, _, _, sc = self._run(monkeypatch, no_save_attention=False)
        assert sc["attention"].shape == (2, 256, 2)      # narrowed to inst_cols

    def test_attention_is_dropped_with_the_flag(self, monkeypatch):
        _, _, _, sc = self._run(monkeypatch, no_save_attention=True)
        assert sc["attention"] is None
        assert sc["cell_attention"] is None

    def test_instance_predictions_are_unaffected(self, monkeypatch):
        _, _, _, with_attn = self._run(monkeypatch, no_save_attention=False)
        _, _, _, without = self._run(monkeypatch, no_save_attention=True)
        np.testing.assert_array_equal(without["instance"], with_attn["instance"])
        assert without["instance"].shape == (2, 256, 2)

    def test_bag_predictions_are_unaffected(self, monkeypatch):
        _, with_attn, _, _ = self._run(monkeypatch, no_save_attention=False)
        _, without, _, _ = self._run(monkeypatch, no_save_attention=True)
        np.testing.assert_array_equal(without, with_attn)

    def test_the_memory_guard_lets_through_what_it_would_have_refused(self, monkeypatch):
        # 2 tiles x 2 targets is nowhere near any limit, so pin the guard directly:
        # the estimate the flag feeds it is the halved one.
        assert predict_he._instance_store_gb(2, 2, True, False) < \
            predict_he._instance_store_gb(2, 2, True, True)

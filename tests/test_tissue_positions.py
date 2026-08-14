"""Seam: reading a sample's spot table off disk.

The four skin cohorts write three different variants of the same table, and every
scale calculation in the pipeline starts by parsing it. Expected values here are
hand-written literals, not re-derived from the loader.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spamil.io import load_tissue_positions

COLUMNS = ["barcode", "in_tissue", "array_row", "array_col",
           "pxl_row_in_fullres", "pxl_col_in_fullres"]

ROWS = [
    ("AAACAACGAATAGTTC-1", 1, 0, 0, 1000, 2000),
    ("AAACAAGTATCTCCCA-1", 1, 0, 2, 1000, 2400),
    ("AAACAATCTACTAGCA-1", 0, 1, 1, 1346, 2200),
]


def _write(tmp_path, name, header):
    spatial = tmp_path / "spatial"
    spatial.mkdir(exist_ok=True)
    df = pd.DataFrame(ROWS, columns=COLUMNS)
    df.to_csv(spatial / name, index=False, header=header)
    return tmp_path


class TestLoadTissuePositions:
    def test_reads_headerless_tissue_positions_list(self, tmp_path):
        vdir = _write(tmp_path, "tissue_positions_list.csv", header=False)
        pos = load_tissue_positions(vdir)
        assert list(pos.columns) == COLUMNS
        assert len(pos) == 3
        assert pos.loc[0, "barcode"] == "AAACAACGAATAGTTC-1"
        assert pos.loc[1, "pxl_col_in_fullres"] == 2400

    def test_reads_tissue_positions_list_that_has_a_header(self, tmp_path):
        # Some datasets ship a *_list.csv WITH a header row anyway, so the
        # filename cannot be trusted to imply the format.
        vdir = _write(tmp_path, "tissue_positions_list.csv", header=True)
        pos = load_tissue_positions(vdir)
        assert len(pos) == 3, "the header row must not be parsed as data"
        assert pos.loc[0, "barcode"] == "AAACAACGAATAGTTC-1"
        assert pos.loc[2, "array_row"] == 1

    def test_prefers_modern_tissue_positions_over_the_legacy_list(self, tmp_path):
        # HEST ships both; the modern file is authoritative.
        spatial = tmp_path / "spatial"
        spatial.mkdir()
        pd.DataFrame(ROWS, columns=COLUMNS).to_csv(
            spatial / "tissue_positions.csv", index=False)
        pd.DataFrame(ROWS[:1], columns=COLUMNS).to_csv(
            spatial / "tissue_positions_list.csv", index=False, header=False)
        assert len(load_tissue_positions(tmp_path)) == 3

    def test_coordinate_columns_are_numeric(self, tmp_path):
        vdir = _write(tmp_path, "tissue_positions_list.csv", header=True)
        pos = load_tissue_positions(vdir)
        for col in ("array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"):
            assert pd.api.types.is_numeric_dtype(pos[col]), col

    def test_raises_when_no_positions_file_exists(self, tmp_path):
        (tmp_path / "spatial").mkdir()
        with pytest.raises(FileNotFoundError):
            load_tissue_positions(tmp_path)

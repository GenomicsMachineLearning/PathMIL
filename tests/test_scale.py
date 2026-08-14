"""Seam: physical scale of a slide, and the patch size that follows from it.

Slide MPP is `100 um / spot pitch in pixels`, deliberately *not*
`55 / spot_diameter_fullres`. These tests pin that against hand-constructed grids
of known geometry, built independently of anything in `spamil.scale`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spamil.scale import (
    native_read_size,
    resolve_patch_geometry,
    slide_mpp_from_positions,
    spot_pitch_px,
)

def hex_grid(pitch_px, n_rows=8, n_cols=12, rotation_deg=0.0, origin=(0.0, 0.0)):
    """A Visium-like hex grid: in-row neighbours differ by array_col 2, `pitch_px` apart.

    Built from the slide's geometry (row spacing = pitch * sin(60 deg)), not from
    anything in spamil.scale.
    """
    theta = math.radians(rotation_deg)
    rows = []
    for r in range(n_rows):
        for c in range(r % 2, 2 * n_cols, 2):
            x = (c / 2.0) * pitch_px
            y = r * pitch_px * math.sin(math.radians(60))
            xr = origin[0] + x * math.cos(theta) - y * math.sin(theta)
            yr = origin[1] + x * math.sin(theta) + y * math.cos(theta)
            rows.append((f"BC{r}_{c}-1", 1, r, c, yr, xr))
    return pd.DataFrame(rows, columns=[
        "barcode", "in_tissue", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres"])


class TestSpotPitch:
    def test_measures_the_in_row_neighbour_distance(self):
        assert spot_pitch_px(hex_grid(400.0)) == pytest.approx(400.0)

    def test_is_invariant_under_rotation(self):
        # A rigid registration rotates the grid; the physical pitch cannot change.
        assert spot_pitch_px(hex_grid(217.3, rotation_deg=37.0)) == pytest.approx(217.3)

    def test_is_invariant_under_translation(self):
        assert spot_pitch_px(hex_grid(400.0, origin=(15000.0, -230.0))) == pytest.approx(400.0)

    def test_ignores_pairs_that_are_not_in_row_neighbours(self):
        # Drop every third spot so some in-row gaps span array_col 4 (200 um).
        grid = hex_grid(400.0, n_rows=6, n_cols=12)
        thinned = grid.iloc[[i for i in range(len(grid)) if i % 3 != 1]].reset_index(drop=True)
        assert spot_pitch_px(thinned) == pytest.approx(400.0)

    def test_tolerates_a_few_corrupt_coordinates(self):
        grid = hex_grid(400.0)
        grid.loc[3, "pxl_col_in_fullres"] = np.nan
        grid.loc[7, "pxl_row_in_fullres"] = np.nan
        assert spot_pitch_px(grid) == pytest.approx(400.0)

    def test_raises_when_no_in_row_neighbours_exist(self):
        single = hex_grid(400.0).iloc[[0]].reset_index(drop=True)
        with pytest.raises(ValueError):
            spot_pitch_px(single)


class TestSlideMpp:
    def test_pitch_of_400px_is_a_quarter_micron_per_pixel(self):
        # 100 um between spot centres over 400 px.
        assert slide_mpp_from_positions(hex_grid(400.0)) == pytest.approx(0.25)

    def test_coarser_slides_give_larger_mpp(self):
        assert slide_mpp_from_positions(hex_grid(75.0)) == pytest.approx(100.0 / 75.0)


class TestNativeReadSize:
    def test_a_slide_already_at_target_reads_the_model_patch_size(self):
        assert native_read_size(0.2535, 0.2535, 224) == 224

    def test_coarser_slides_read_fewer_native_pixels(self):
        # Hand-worked: round(224 * 0.2535 / slide_mpp), across a 5.3x
        # magnification range.
        assert native_read_size(0.8318, 0.2535, 224) == 68
        assert native_read_size(1.3332, 0.2535, 224) == 43 
        assert native_read_size(0.5051, 0.2535, 224) == 112
        assert native_read_size(0.2519, 0.2535, 224) == 225

    def test_finer_slides_read_more_native_pixels(self):
        assert native_read_size(0.1268, 0.2535, 224) == 448

    def test_is_never_below_one_pixel(self):
        assert native_read_size(1e6, 0.2535, 224) == 1

    def test_rejects_a_non_positive_slide_mpp(self):
        with pytest.raises(ValueError):
            native_read_size(0.0, 0.2535, 224)


def _sample_on_disk(tmp_path, library_id, pitch_px):
    vdir = tmp_path / library_id
    (vdir / "spatial").mkdir(parents=True)
    hex_grid(pitch_px).to_csv(
        vdir / "spatial" / "tissue_positions_list.csv", index=False, header=False)
    return tmp_path


def _cfg(data_dir, target_mpp, patch_size=224, image_patch_width=224):
    return {
        "dataset": {"data_dir": str(data_dir), "layout": "flat"},
        "embed": {"patch_size": patch_size,
                  "image_patch_width": image_patch_width,
                  "target_mpp": target_mpp},
    }


class TestResolvePatchGeometry:
    def test_target_mpp_unset_keeps_the_historical_native_pixel_behaviour(self, tmp_path):
        # Every pre-existing config and checkpoint depends on this staying identical.
        root = _sample_on_disk(tmp_path, "LIB1", pitch_px=120.2)
        geom = resolve_patch_geometry("LIB1", _cfg(root, None, 224, 448))
        assert geom["target_mpp"] is None
        assert geom["slide_mpp"] is None
        assert geom["spot_read_px"] == 224
        assert geom["image_read_px"] == 448

    def test_target_mpp_set_reads_native_pixels_from_the_measured_pitch(self, tmp_path):
        # pitch 120.2 px -> 0.8319 um/px (Sam_aligned_v3's scale) -> 68 px at 0.2535.
        root = _sample_on_disk(tmp_path, "PMP_B1", pitch_px=120.2)
        geom = resolve_patch_geometry("PMP_B1", _cfg(root, 0.2535))
        assert geom["slide_mpp"] == pytest.approx(0.8319, abs=1e-4)
        assert geom["spot_read_px"] == 68
        assert geom["fov_um"] == pytest.approx(56.784)

    def test_the_field_of_view_is_the_same_micrometres_for_every_slide(self, tmp_path):
        # The whole point of target_mpp: different pixel reads, identical
        # tissue extent.
        root = _sample_on_disk(tmp_path, "FINE", pitch_px=396.0)
        _sample_on_disk(root, "COARSE", pitch_px=75.0)
        fine = resolve_patch_geometry("FINE", _cfg(root, 0.2535))
        coarse = resolve_patch_geometry("COARSE", _cfg(root, 0.2535))
        assert fine["spot_read_px"] != coarse["spot_read_px"]
        assert fine["fov_um"] == coarse["fov_um"] == pytest.approx(56.784)

    def test_the_image_grid_scales_with_its_own_configured_width(self, tmp_path):
        root = _sample_on_disk(tmp_path, "LIB1", pitch_px=120.2)
        geom = resolve_patch_geometry("LIB1", _cfg(root, 0.2535, 224, 448))
        # Each width rounds independently: 224*0.2535/0.831947 = 68.25 -> 68, and
        # 448*0.2535/0.831947 = 136.51 -> 137. The image grid is deliberately not
        # constrained to an exact multiple of the spot read.
        assert geom["spot_read_px"] == 68
        assert geom["image_read_px"] == 137

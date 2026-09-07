"""Physical scale of a Visium slide, and the patch geometry that follows.

`preprocess` cuts a fixed-size square around each spot and hands it to the
foundation model. For that square to mean the same thing across slides it has to
cover a fixed number of *micrometres*, not a fixed number of pixels. Scanners
disagree by a lot: a cohort spanning 0.25-1.33 um/px is a 5.3x range, so one
fixed 224 px patch would be 56.7 um of tissue on the finest slide and 298.6 um on
the coarsest -- different objects entirely, presented to the model as if they
were the same.

Converting between the two needs the slide's micrometres-per-pixel. Two obvious
sources are both unreliable:

  * `55 / spot_diameter_fullres` -- Space Ranger's `spot_diameter_fullres` does
    not measure the 55 um capture spot, and what it does measure is not even a
    fixed convention: across four cohorts it works out at 70.0, 71.0, 65.0 and
    59.9 um against the spot pitch, varying by Space Ranger version. Treating it
    as 55 um under-estimates MPP by 9-29%.
  * TIFF resolution tags -- frequently placeholders (300 dpi, 96 dpi,
    1 px = 1 mm) rather than the real scan resolution.

What is reliable is the slide's spot pitch: two spots adjacent in the same array
row are 100 um apart by hardware. It is expressed in the same coordinate space as
`pxl_*_in_fullres` (the space patches are actually cut in) and survives any rigid
or affine registration.
"""

from __future__ import annotations

import numpy as np

# Centre-to-centre distance between spots adjacent in an array row, in micrometres.
# Fixed by the Visium slide for v1, v2 and CytAssist capture areas.
SPOT_PITCH_UM = 100.0

# Spots adjacent within a row differ by this much in `array_col` (the hex grid
# interleaves odd and even rows, so consecutive columns are 2 apart).
IN_ROW_COL_STEP = 2


def spot_pitch_px(positions) -> float:
    """Median pixel distance between spots adjacent in the same array row.

    Args:
        positions: frame with `array_row`, `array_col`, `pxl_row_in_fullres`,
            `pxl_col_in_fullres` (as returned by `pathmil.io.load_tissue_positions`).

    Uses the median so a handful of misplaced spots cannot move the estimate, and
    the full 2-D distance so the answer is rotation-invariant.
    """
    needed = ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos = positions.dropna(subset=needed)

    distances = []
    for _, row_spots in pos.groupby("array_row"):
        row_spots = row_spots.sort_values("array_col")
        col = row_spots["array_col"].to_numpy()
        x = row_spots["pxl_col_in_fullres"].to_numpy(dtype=float)
        y = row_spots["pxl_row_in_fullres"].to_numpy(dtype=float)
        adjacent = np.diff(col) == IN_ROW_COL_STEP
        if adjacent.any():
            distances.append(np.hypot(np.diff(x)[adjacent], np.diff(y)[adjacent]))

    if not distances:
        raise ValueError(
            "No in-row neighbouring spots found; cannot measure the spot pitch. "
            "Expected pairs of spots sharing an array_row with array_col differing "
            f"by {IN_ROW_COL_STEP}.")

    pitch = float(np.median(np.concatenate(distances)))
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError(f"Measured a non-physical spot pitch: {pitch} px")
    return pitch


def slide_mpp_from_positions(positions, pitch_um: float = SPOT_PITCH_UM) -> float:
    """Micrometres per pixel of the full-resolution image, from the spot grid."""
    return pitch_um / spot_pitch_px(positions)


def slide_mpp(library_id: str, cfg: dict) -> float:
    """Micrometres per pixel for one sample, read from its `spatial/` directory."""
    from pathmil import io as sio

    return slide_mpp_from_positions(
        sio.load_tissue_positions(sio.visium_dir(library_id, cfg)))


def native_read_size(slide_mpp_value: float, target_mpp: float,
                     patch_size: int = 224) -> int:
    """How many native pixels to read so that, resized to `patch_size`, the patch
    covers `patch_size * target_mpp` micrometres of tissue.

    Returns `patch_size` exactly when the slide is already at the target scale.
    """
    if not slide_mpp_value or slide_mpp_value <= 0:
        raise ValueError(f"slide_mpp must be positive, got {slide_mpp_value!r}")
    if not target_mpp or target_mpp <= 0:
        raise ValueError(f"target_mpp must be positive, got {target_mpp!r}")
    return max(1, int(round(patch_size * target_mpp / slide_mpp_value)))


def resolve_patch_geometry(library_id: str, cfg: dict) -> dict:
    """Patch geometry for one sample: what to read natively, and why.

    `embed.target_mpp` unset (null) keeps the historical behaviour exactly -- read
    `embed.patch_size` native pixels and do no rescaling -- so every pre-existing
    config and checkpoint is unaffected.
    """
    from pathmil.config import cget

    patch_size = int(cget(cfg, "embed.patch_size", 224))
    image_patch_width = int(cget(cfg, "embed.image_patch_width", patch_size))
    target_mpp = cget(cfg, "embed.target_mpp")

    if target_mpp is None:
        return {
            "target_mpp": None,
            "slide_mpp": None,
            "patch_size": patch_size,
            "spot_read_px": patch_size,
            "image_read_px": image_patch_width,
            "fov_um": None,
        }

    target_mpp = float(target_mpp)
    mpp = slide_mpp(library_id, cfg)
    return {
        "target_mpp": target_mpp,
        "slide_mpp": mpp,
        "patch_size": patch_size,
        "spot_read_px": native_read_size(mpp, target_mpp, patch_size),
        "image_read_px": native_read_size(mpp, target_mpp, image_patch_width),
        "fov_um": patch_size * target_mpp,
    }

#!/usr/bin/env python
"""Predict gene / gene-module scores directly from an H&E slide.

One script, three stages, no intermediate files:

    slide -> tissue tiles -> Virchow2 patch tokens -> MIL regressor -> per-tile scores

Each tissue tile is one MIL bag: Virchow2 returns 256 spatial tokens per 224px tile,
which is exactly the `(n_bags, 256, 1280)` input the PathMIL regressor was trained on.
Tiles are streamed straight from the foundation model into the MIL head, so the
`(n_tiles, 256, 1280)` float32 embedding array -- tens of GB on a whole-slide image --
is never materialized. Pass --save-embeddings if you want it on disk anyway.

The checkpoint is self-describing: it carries the architecture *and* the target names,
so the output columns are labelled with the real gene / module names automatically.

Field of view matters. The model sees a fixed 224px tile, so a slide scanned at a
different magnification than the training data presents the wrong physical FOV unless
you rescale. Pass --target-mpp with the microns-per-pixel the model was TRAINED at;
each tile is then read at the size that covers the same physical area and resized to
224px. Omit it only if your slide is already at the training resolution.

With --sc_pred the script additionally predicts at two finer scales. The MIL head is
additive (the bag score is the attention-weighted sum of its 256 per-instance scores),
so those 256 tokens are a real 16x16 prediction map *inside* each tile. Each tile is
also segmented with StarDist, and every nucleus is scored from the tokens it overlaps:

    tile (bag)  ->  instance (16x16 tokens)  ->  single cell (StarDist nuclei)

Segmentation runs per tile on a slightly padded crop, and a nucleus is kept only by the
tile whose core contains its centroid. Tile ownership is therefore a partition: cells
straddling a tile seam are detected twice but kept once, so nothing is double-counted.

Under a per-gene-attention checkpoint the attention carries a target axis, making it as
large as the instance scores it sits beside. --no-save-attention drops it -- from memory
as well as from disk -- and leaves every prediction untouched.

Example
-------
    python scripts/predict_he.py \
        --slide tumour.svs \
        --checkpoint model_checkpoint.pth \
        --out results/ \
        --target-mpp 0.2535 \
        --sc_pred
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import triton
# The script lives in scripts/; make the sibling package importable when run from a
# source checkout that has not been pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathmil.model import load_checkpoint  # noqa: E402
from pathmil.utils import get_logger  # noqa: E402

log = get_logger()

IMAGE_EXTS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".bif", ".png", ".jpg", ".jpeg",
}


def _safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(name).strip())
    return name.strip("_") or "sample"


# --------------------------------------------------------------------------- slide io


class SlideReader:
    """OpenSlide-first slide reader, falling back to PIL then tifffile.

    Exposes the level-0 dimensions and, when the format records it, the
    microns-per-pixel needed for field-of-view matching.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._openslide = None
        self._pil = None
        self.mpp = None

        try:
            import openslide
            self._openslide = openslide.OpenSlide(str(self.path))
            self.width, self.height = self._openslide.dimensions
            try:
                mx = self._openslide.properties.get(openslide.PROPERTY_NAME_MPP_X)
                self.mpp = float(mx) if mx else None
            except Exception:
                self.mpp = None
            return
        except ImportError:
            openslide_err = (
                "openslide is not installed. Install it with "
                "`pip install openslide-python` (which also needs the OpenSlide C "
                "library: `conda install -c conda-forge openslide-python`, or "
                "`apt install libopenslide0`)."
            )
        except Exception as exc:
            openslide_err = f"OpenSlide could not open the file: {exc}"

        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            self._pil = Image.open(self.path).convert("RGB")
            self.width, self.height = self._pil.size
            return
        except Exception:
            pass

        try:
            import tifffile
            from PIL import Image
            arr = tifffile.imread(self.path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                self._pil = Image.fromarray(arr[..., :3]).convert("RGB")
                self.width, self.height = self._pil.size
                return
        except Exception:
            pass

        raise RuntimeError(f"Could not open {self.path}. {openslide_err}")

    def read_region(self, x: int, y: int, size: int):
        if self._openslide is not None:
            return self._openslide.read_region((int(x), int(y)), 0, (size, size)).convert("RGB")
        return self._pil.crop((int(x), int(y), int(x + size), int(y + size))).convert("RGB")

    def thumbnail(self, max_dim: int):
        if self._openslide is not None:
            return np.asarray(self._openslide.get_thumbnail((max_dim, max_dim)).convert("RGB"))
        thumb = self._pil.copy()
        thumb.thumbnail((max_dim, max_dim))
        return np.asarray(thumb.convert("RGB"))

    def close(self):
        if self._openslide is not None:
            self._openslide.close()
        if self._pil is not None:
            self._pil.close()


def _tile_coords(width: int, height: int, tile_size: int, stride: int) -> list:
    if width < tile_size or height < tile_size:
        return [(0, 0)]
    return [(x, y)
            for y in range(0, height - tile_size + 1, stride)
            for x in range(0, width - tile_size + 1, stride)]


def _tissue_fraction(img, white_threshold: int) -> float:
    arr = np.asarray(img)
    if arr.ndim != 3:
        return 0.0
    return float(np.any(arr < white_threshold, axis=2).mean())


# --------------------------------------------------------------------- segmentation

TOKENS_PER_SIDE = 16  # Virchow2 gives a 16x16 spatial token grid per 224px tile (=256)


def _tf_cpu_only():
    """Hide the GPU from TensorFlow so StarDist runs on CPU.

    The GPU is reserved for the Virchow2 (torch) pass. TensorFlow otherwise claims the
    whole device on first use and the two libraries fight over it. Must be called before
    TF initialises any GPU.
    """
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _load_stardist(model_dir=None):
    """Load StarDist 2D_versatile_he (the H&E nuclei model), pinned to CPU."""
    _tf_cpu_only()
    try:
        from stardist.models import StarDist2D
    except ImportError as exc:
        raise ImportError(
            "--sc_pred needs StarDist. Install the H&E extra: `pip install -e .[he]` "
            "(brings in stardist, csbdeep, tensorflow, shapely, geopandas)."
        ) from exc
    _tf_cpu_only()
    if model_dir:
        return StarDist2D(None, name="2D_versatile_he", basedir=str(model_dir))
    return StarDist2D.from_pretrained("2D_versatile_he")


def _segment_crop(seg_model, crop_rgb, prob_thresh, nms_thresh):
    """Run StarDist on one RGB crop; return (centroids_xy, polygons_xy, areas).

    `details["points"]` are (y, x) centroids and `details["coord"]` the per-nucleus ray
    polygons, so boundaries come straight out of the model -- no contour tracing.
    """
    from csbdeep.utils import normalize

    img = normalize(np.asarray(crop_rgb), clip=True)
    labels, details = seg_model.predict_instances(
        img, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

    pts = np.asarray(details["points"], dtype="float64")          # (n, 2) as (y, x)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2)), [], np.zeros(0)

    centroids = pts[:, ::-1].copy()                               # -> (x, y)
    coord = np.asarray(details["coord"], dtype="float64")         # (n, 2, n_rays) as (y, x)
    polygons = [np.stack([c[1], c[0]], axis=1) for c in coord]    # each (n_rays, 2) as (x, y)
    # Instance i owns label id i+1. Index by instance rather than by labels.max(): the
    # label image can carry a higher max id than there are returned points, and zipping
    # the two lengths together is a silent mis-assignment (it raised on a 65-vs-64 slide).
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    areas = counts[1:n + 1].astype("float64")
    return centroids, polygons, areas


def _tokens_for_bbox(x0, y0, x1, y1, tile_x, tile_y, step):
    """Token indices (into the flat 256) whose cells overlap a bbox, in global px.

    Mirrors the internal `sjoin(..., predicate="intersects") -> groupby.mean`, but as a
    direct index lookup: the token grid is regular, so the overlapping tokens are just a
    row/column range.
    """
    c0 = int(np.clip((x0 - tile_x) // step, 0, TOKENS_PER_SIDE - 1))
    c1 = int(np.clip((x1 - tile_x) // step, 0, TOKENS_PER_SIDE - 1))
    r0 = int(np.clip((y0 - tile_y) // step, 0, TOKENS_PER_SIDE - 1))
    r1 = int(np.clip((y1 - tile_y) // step, 0, TOKENS_PER_SIDE - 1))
    rows = np.arange(r0, r1 + 1)
    cols = np.arange(c0, c1 + 1)
    return (rows[:, None] * TOKENS_PER_SIDE + cols[None, :]).ravel()


# ------------------------------------------------------------------------ inference


def _load_virchow(model_name: str, device):
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        model_name, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
    ).eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def _fov_geometry(reader: SlideReader, tile_size: int, stride: int, target_mpp):
    """Return (read_size, native_stride) so each tile covers the training FOV.

    The model always consumes `tile_size` px. When the slide's resolution differs from
    the one the model was trained at, read a correspondingly larger/smaller region and
    resample it to `tile_size`; the stride is scaled by the same factor so coverage
    stays consistent.
    """
    if target_mpp and reader.mpp:
        scale = target_mpp / reader.mpp
        read_size = max(1, int(round(tile_size * scale)))
        native_stride = max(1, int(round(stride * scale)))
        log.info("FOV-match: slide_mpp=%.4f target_mpp=%.4f -> read %dpx and resize to "
                 "%dpx (stride %dpx)", reader.mpp, target_mpp, read_size, tile_size,
                 native_stride)
        return read_size, native_stride

    if target_mpp and not reader.mpp:
        log.warning("--target-mpp given but the slide does not report its MPP; tiling "
                    "natively at %dpx. Predictions may be at the wrong field of view.",
                    tile_size)
    return tile_size, stride


def _instance_store_gb(n_tiles, n_targets, per_gene, keep_attention):
    """GB the --sc_pred stores need: the instance array, plus attention if it is kept.

    Per-gene attention is narrowed to the same columns as the instances, so it is a
    second array of identical shape -- the dominant term alongside the scores. Shared
    attention is (n_tiles, 256) and negligible beside either, and --no-save-attention
    drops the term entirely.
    """
    n_stores = 2 if (per_gene and keep_attention) else 1
    return n_stores * n_tiles * TOKENS_PER_SIDE ** 2 * n_targets * 4 / 1e9


def predict_slide(slide_path: Path, reader: SlideReader, virchow, transform, model, args,
                  device, inst_cols=None, seg_model=None, emb_h5_path: Path | None = None):
    """Stream one slide through Virchow2 -> MIL; return (tile_xy, bag_preds, attrs, sc).

    Only the batch currently in flight is held in memory. When `emb_h5_path` is given
    the raw tokens are appended to a resizable H5 dataset as they are computed, so even
    --save-embeddings never accumulates the whole slide in RAM.

    With `seg_model` (i.e. --sc_pred) each tile is additionally segmented with StarDist
    and every nucleus is scored from the MIL instance head; `sc` then carries the
    per-tile instance predictions/attention and the per-cell table. Otherwise `sc` is
    None and the behaviour is byte-for-byte the tile-level path.

    Under --no-save-attention the attention is never accumulated -- not merely left
    unwritten -- so `sc["attention"]` and `sc["cell_attention"]` are None and the peak
    memory really does drop. Everything else is unchanged.
    """
    import torch
    from PIL import Image

    log.info("[%s] %d x %d px, mpp=%s", slide_path.name, reader.width, reader.height,
             f"{reader.mpp:.4f}" if reader.mpp else "unknown")

    read_size, native_stride = _fov_geometry(reader, args.tile_size, args.stride,
                                             args.target_mpp)
    coords = _tile_coords(reader.width, reader.height, read_size, native_stride)
    log.info("[%s] %d candidate tiles", slide_path.name, len(coords))

    sc_pred = seg_model is not None
    keep_attn = sc_pred and not args.no_save_attention
    token_step = read_size / TOKENS_PER_SIDE   # native px covered by one Virchow2 token
    margin = int(args.seg_margin) if sc_pred else 0
    crop_size = read_size + 2 * margin

    if sc_pred:
        # Upper bound: every candidate tile kept. The instance array is what makes
        # --sc_pred expensive, so refuse up front rather than dying mid-slide.
        per_gene = getattr(model, "attention_mode", "shared") == "per_gene"
        both = per_gene and keep_attn
        need_gb = _instance_store_gb(len(coords), len(inst_cols), per_gene, keep_attn)
        if need_gb > args.max_instance_gb:
            raise MemoryError(
                f"--sc_pred would need up to {need_gb:.1f} GB to hold instance "
                f"predictions{' and per-gene attention' if both else ''} "
                f"for {len(inst_cols)} targets over {len(coords)} tiles "
                f"(limit --max-instance-gb={args.max_instance_gb}). Pass --targets to "
                f"score only the targets you need, "
                f"{'--no-save-attention to drop the attention half, ' if both else ''}"
                f"or raise --max-instance-gb.")
        log.info("[%s] --sc_pred: token step %.2f px, seg margin %d px, %d targets; "
                 "instance%s store <= %.1f GB", slide_path.name, token_step, margin,
                 len(inst_cols), "+attention" if both else "", need_gb)
        if not keep_attn:
            log.info("[%s] --no-save-attention: attention is neither accumulated nor "
                     "written", slide_path.name)

    kept: list = []
    preds: list = []
    inst_all: list = []
    attn_all: list = []
    batch: list = []
    batch_xy: list = []
    batch_crops: list = []

    # Per-cell records, accumulated across tiles.
    cell_xy: list = []
    cell_area: list = []
    cell_poly: list = []
    cell_score: list = []
    cell_attn: list = []

    emb_file = emb_ds = None
    if emb_h5_path is not None:
        import h5py
        emb_file = h5py.File(emb_h5_path, "w")
        emb_ds = emb_file.create_dataset(
            "embeddings", shape=(0, 256, 1280), maxshape=(None, 256, 1280),
            dtype="float32", chunks=(max(1, min(args.batch_size, 32)), 256, 1280))

    def flush():
        if not batch:
            return
        x = torch.cat(batch, dim=0).to(device)
        with torch.no_grad():
            # Virchow2 returns [CLS] + 4 register tokens + 256 spatial tokens; the MIL
            # bag is the 256 spatial tokens only. The slice is a non-contiguous view and
            # the regressor reshapes with .view(), so make it contiguous first.
            tokens = virchow(x)[:, 5:].contiguous()       # (B, 256, 1280)
            if sc_pred:
                bag, inst, attn = model(tokens.float(), return_instance_predictions=True)
            else:
                bag = model(tokens.float())               # (B, n_outputs)
        preds.append(bag.cpu().numpy())
        if emb_ds is not None:
            arr = tokens.cpu().float().numpy()
            start = emb_ds.shape[0]
            emb_ds.resize((start + arr.shape[0], 256, 1280))
            emb_ds[start:start + arr.shape[0]] = arr

        if sc_pred:
            inst_np = inst.cpu().numpy()[:, :, inst_cols]  # (B, 256, n_targets)
            attn_np = None
            if keep_attn:
                # (B, 256) under shared attention; (B, 256, n_outputs) under per-gene,
                # where it must be narrowed by the SAME columns as the instances --
                # otherwise the attention store stays at the full target set, which is
                # both the dominant memory term and a different set of targets from
                # the scores sitting next to it.
                attn_np = attn.cpu().numpy()
                if attn_np.ndim == 3:
                    attn_np = attn_np[:, :, inst_cols]     # (B, 256, n_targets)
                attn_all.append(attn_np)
            inst_all.append(inst_np)
            for b, ((tx, ty), (ox, oy, crop)) in enumerate(zip(batch_xy, batch_crops)):
                _score_cells_in_tile(seg_model, crop, ox, oy, tx, ty, read_size,
                                     token_step, inst_np[b],
                                     attn_np[b] if attn_np is not None else None, args,
                                     cell_xy, cell_area, cell_poly, cell_score, cell_attn)

        kept.extend(batch_xy)
        batch.clear()
        batch_xy.clear()
        batch_crops.clear()

    try:
        n_seen = 0
        for x, y in coords:
            if sc_pred:
                # One read per tile: take the padded crop (clamped to stay inside the
                # slide, since OpenSlide returns transparent -> black outside) and slice
                # the tile out of its core rather than reading twice.
                ox = int(np.clip(x - margin, 0, max(0, reader.width - crop_size)))
                oy = int(np.clip(y - margin, 0, max(0, reader.height - crop_size)))
                crop = reader.read_region(ox, oy, crop_size)
                cx, cy = x - ox, y - oy
                img = crop.crop((cx, cy, cx + read_size, cy + read_size))
            else:
                crop = None
                img = reader.read_region(x, y, read_size)

            if read_size != args.tile_size:
                tile_img = img.resize((args.tile_size, args.tile_size), Image.LANCZOS)
            else:
                tile_img = img
            if _tissue_fraction(tile_img, args.white_threshold) < args.min_tissue:
                continue

            batch.append(transform(tile_img).unsqueeze(0))
            batch_xy.append((x, y))
            if sc_pred:
                batch_crops.append((ox, oy, crop))
            n_seen += 1
            if len(batch) >= args.batch_size:
                flush()
                log.info("[%s] %d tiles done%s", slide_path.name, len(kept),
                         f", {len(cell_xy)} cells" if sc_pred else "")
            if args.max_tiles and n_seen >= args.max_tiles:
                break
        flush()

        if not kept:
            raise RuntimeError(
                f"No tissue tiles found in {slide_path.name}; try lowering --min-tissue "
                f"(currently {args.min_tissue}) or raising --white-threshold.")

        tile_xy = np.asarray(kept, dtype="int64")
        bag_preds = np.concatenate(preds, axis=0)
        attrs = {
            "image_width": reader.width,
            "image_height": reader.height,
            "tile_size": read_size,
            "patch_size": args.tile_size,
            "stride": native_stride,
            "token_step": token_step,
            "slide_mpp": reader.mpp or 0.0,
            "target_mpp": args.target_mpp or 0.0,
        }
        log.info("[%s] %d tissue tiles x %d outputs", slide_path.name, *bag_preds.shape)

        sc = None
        if sc_pred:
            xy = np.asarray(cell_xy, dtype="float64").reshape(-1, 2)
            keep = _dedup_seam_cells(xy, args.dedup_radius)
            if len(keep) < len(xy):
                log.info("[%s] dropped %d seam-duplicate nuclei (%.2f%%)",
                         slide_path.name, len(xy) - len(keep),
                         100 * (len(xy) - len(keep)) / max(1, len(xy)))
            sc = {
                "instance": np.concatenate(inst_all, axis=0),     # (n_tiles, 256, n_tgt)
                # None under --no-save-attention; otherwise (n_tiles, 256) shared or
                # (n_tiles, 256, n_tgt) per-gene.
                "attention": np.concatenate(attn_all, axis=0) if keep_attn else None,
                "cell_xy": xy[keep],
                "cell_area": np.asarray(cell_area, dtype="float32")[keep],
                "cell_score": np.asarray(cell_score, dtype="float32").reshape(
                    -1, len(inst_cols))[keep],
                "cell_attention": (np.asarray(cell_attn, dtype="float32")[keep]
                                   if keep_attn else None),
                "cell_poly": [cell_poly[i] for i in keep],
            }
            log.info("[%s] segmented %d nuclei (%.1f per tile)", slide_path.name,
                     len(keep), len(keep) / max(1, len(tile_xy)))

        if emb_file is not None:
            emb_file.create_dataset("tile_xy", data=tile_xy)
            for k, v in attrs.items():
                emb_file.attrs[k] = v
            emb_file.attrs["n_spots"] = len(tile_xy)
            emb_file.attrs["sample_id"] = _safe(slide_path.stem)
            emb_file.attrs["source_image"] = str(slide_path)
        return tile_xy, bag_preds, attrs, sc
    finally:
        if emb_file is not None:
            emb_file.close()


def _dedup_seam_cells(xy, radius):
    """Drop nuclei detected twice on a tile seam. Returns the indices to keep.

    Centroid-in-core ownership removes almost all seam duplication, but it silently
    assumes both neighbouring tiles estimate the *same* centroid for a nucleus lying on
    their shared edge. They do not: StarDist sees different context in each padded crop,
    so a nucleus within ~1px of the boundary can be placed just inside core A by tile A
    and just inside core B by tile B, and survive in both (~1% of nuclei in practice).

    Two genuinely distinct nuclei are never this close -- they would have to overlap --
    so collapsing centroids within `radius` px removes exactly the seam artefacts.
    """
    if radius <= 0 or len(xy) < 2:
        return np.arange(len(xy))
    from scipy.spatial import cKDTree

    pairs = cKDTree(xy).query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(xy))
    drop = np.zeros(len(xy), dtype=bool)
    drop[np.maximum(pairs[:, 0], pairs[:, 1])] = True   # keep the first of each pair
    return np.nonzero(~drop)[0]


def _score_cells_in_tile(seg_model, crop, ox, oy, tile_x, tile_y, read_size, token_step,
                         inst, attn, args, cell_xy, cell_area, cell_poly, cell_score,
                         cell_attn):
    """Segment one padded crop and score the nuclei this tile owns.

    Tile ownership is a *partition*: a nucleus is kept only by the tile whose core rect
    contains its centroid. Nuclei straddling a seam are detected in both neighbours but
    survive in exactly one, so there is no double counting and no NMS pass is needed.
    """
    centroids, polys, areas = _segment_crop(seg_model, crop, args.seg_prob_thresh,
                                            args.seg_nms_thresh)
    if len(centroids) == 0:
        return

    # Crop-local -> global px.
    gx = centroids[:, 0] + ox
    gy = centroids[:, 1] + oy

    core = ((gx >= tile_x) & (gx < tile_x + read_size) &
            (gy >= tile_y) & (gy < tile_y + read_size) &
            (areas >= args.min_cell_area))
    for i in np.nonzero(core)[0]:
        poly = polys[i] + np.array([ox, oy])            # (n_rays, 2) global (x, y)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        toks = _tokens_for_bbox(x0, y0, x1, y1, tile_x, tile_y, token_step)
        cell_xy.append((gx[i], gy[i]))
        cell_area.append(areas[i])
        cell_poly.append(poly)
        cell_score.append(inst[toks].mean(axis=0))      # mean over overlapping tokens
        if attn is None:                                # --no-save-attention
            continue
        # Shared attention is (256,) -> one weight per nucleus. Per-gene attention
        # is (256, n_targets) -> one weight per nucleus PER TARGET; averaging that
        # down to a scalar would silently mix every target into one number.
        cell_attn.append(attn[toks].mean(axis=0) if attn.ndim == 2
                         else float(attn[toks].mean()))


# -------------------------------------------------------------------------- heatmaps


def _score_grid(tile_xy, values, step, W, H):
    """Place per-tile values on the tile lattice; NaN where no tissue tile was kept."""
    ncol, nrow = int(W // step) + 1, int(H // step) + 1
    grid = np.full((nrow, ncol), np.nan, dtype="float32")
    cols = np.clip(tile_xy[:, 0] // step, 0, ncol - 1).astype(int)
    rows = np.clip(tile_xy[:, 1] // step, 0, nrow - 1).astype(int)
    grid[rows, cols] = values
    return np.ma.masked_invalid(grid)


def _vlim(values):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None, None
    lo, hi = np.percentile(v, [2, 98])
    if lo == hi:
        lo, hi = float(v.min()), float(v.max())
    return float(lo), float(hi)


def _overlay(ax, thumb, grid, W, H, cmap, vmin, vmax, title):
    extent = [0, W, H, 0]  # full-res pixel coords, origin top-left
    alpha = 1.0
    if thumb is not None:
        ax.imshow(thumb, extent=extent, aspect="auto")
        alpha = 0.6
    im = ax.imshow(grid, extent=extent, cmap=cmap, alpha=alpha, interpolation="nearest",
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def write_heatmaps(sample_id, reader, tile_xy, bag, names, attrs, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = int(attrs["stride"])
    W, H = int(attrs["image_width"]), int(attrs["image_height"])
    try:
        thumb = reader.thumbnail(args.thumbnail_max_dim)
    except Exception as exc:
        log.warning("[%s] thumbnail failed (%s); heatmaps will have no H&E backdrop",
                    sample_id, exc)
        thumb = None

    n_out = bag.shape[1]
    ncols = int(math.ceil(math.sqrt(n_out)))
    nrows = int(math.ceil(n_out / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_out):
        vmin, vmax = _vlim(bag[:, i])
        _overlay(axes[i], thumb, _score_grid(tile_xy, bag[:, i], step, W, H),
                 W, H, args.cmap, vmin, vmax, names[i])
    for j in range(n_out, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{sample_id} — {n_out} predicted targets (H&E inference)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / f"{sample_id}_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for i in _select_targets(names, bag, args, sample_id):
        vmin, vmax = _vlim(bag[:, i])
        fig, ax = plt.subplots(figsize=(9, 8))
        im = _overlay(ax, thumb, _score_grid(tile_xy, bag[:, i], step, W, H),
                      W, H, args.cmap, vmin, vmax, f"{sample_id} — {names[i]}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="predicted score")
        fig.savefig(out_dir / f"{sample_id}_{_safe(names[i])}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def _select_targets(names, bag, args, sample_id):
    """Indices of the targets to render: --targets, else the n_top most variable."""
    if args.targets:
        missing = [t for t in args.targets if t not in names]
        if missing:
            log.warning("[%s] not in this checkpoint: %s", sample_id, ", ".join(missing))
        return [names.index(t) for t in args.targets if t in names]
    return [int(i) for i in np.argsort(np.nanvar(bag, axis=0))[::-1][:args.n_top]]


# ------------------------------------------------------- single-cell / instance figures


def _instance_raster(tile_xy, inst, target_idx, token_step, W, H):
    """Rasterize the 16x16 per-tile token grids onto one slide-wide lattice.

    The lattice step is `token_step` native px (= read_size / 16), so this map is 16x
    finer than the tile-level one. NaN wherever no tissue tile covered the cell.
    """
    ncol = int(W // token_step) + 1
    nrow = int(H // token_step) + 1
    grid = np.full((nrow, ncol), np.nan, dtype="float32")
    vals = inst[:, :, target_idx].reshape(-1, TOKENS_PER_SIDE, TOKENS_PER_SIDE)
    for t, (tx, ty) in enumerate(tile_xy):
        c0 = int(tx // token_step)
        r0 = int(ty // token_step)
        c1 = min(c0 + TOKENS_PER_SIDE, ncol)
        r1 = min(r0 + TOKENS_PER_SIDE, nrow)
        grid[r0:r1, c0:c1] = vals[t, :r1 - r0, :c1 - c0]
    return np.ma.masked_invalid(grid)


def write_sc_figures(sample_id, reader, tile_xy, bag, names, inst_names, attrs, sc,
                     out_dir, args):
    """4-panel (H&E+cells | tile | instance | single-cell) + a full-res segmentation QC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = int(attrs["image_width"]), int(attrs["image_height"])
    step = int(attrs["stride"])
    token_step = float(attrs["token_step"])
    cells_xy = sc["cell_xy"]
    cell_score = sc["cell_score"]

    if len(cells_xy) == 0:
        log.warning("[%s] no nuclei segmented; skipping single-cell figures", sample_id)
        return

    try:
        thumb = reader.thumbnail(args.thumbnail_max_dim)
    except Exception as exc:
        log.warning("[%s] thumbnail failed (%s)", sample_id, exc)
        thumb = None

    # Point size: nuclei are sub-pixel at thumbnail scale, so this is a density view.
    psize = max(0.05, 4.0 * (args.thumbnail_max_dim / max(W, H)))

    for i in _select_targets(names, bag, args, sample_id):
        name = names[i]
        if name not in inst_names:
            continue  # not scored at instance level (restricted by --targets)
        j = inst_names.index(name)   # column into the instance / cell arrays
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        # 1. H&E + segmented nuclei
        ax = axes[0, 0]
        if thumb is not None:
            ax.imshow(thumb, extent=[0, W, H, 0], aspect="auto")
        ax.scatter(cells_xy[:, 0], cells_xy[:, 1], s=psize, c="#00d0ff", alpha=0.35,
                   linewidths=0)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f"H&E + {len(cells_xy):,} StarDist nuclei", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # 2. tile (bag) level
        vmin, vmax = _vlim(bag[:, i])
        im = _overlay(axes[0, 1], thumb, _score_grid(tile_xy, bag[:, i], step, W, H),
                      W, H, args.cmap, vmin, vmax, f"{name} — prediction (tile)")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)

        # 3. instance level (16x16 tokens inside every tile)
        ivmin, ivmax = _vlim(sc["instance"][:, :, j].ravel())
        im = _overlay(axes[1, 0], thumb,
                      _instance_raster(tile_xy, sc["instance"], j, token_step, W, H),
                      W, H, args.cmap, ivmin, ivmax,
                      f"{name} — prediction (instance, {token_step:.1f}px tokens)")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.02)

        # 4. single-cell level
        ax = axes[1, 1]
        if thumb is not None:
            ax.imshow(thumb, extent=[0, W, H, 0], aspect="auto")
        cvmin, cvmax = _vlim(cell_score[:, j])
        sc_im = ax.scatter(cells_xy[:, 0], cells_xy[:, 1], s=psize, c=cell_score[:, j],
                           cmap=args.cmap, vmin=cvmin, vmax=cvmax, alpha=0.8, linewidths=0)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f"{name} — prediction (single cell)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc_im, ax=ax, fraction=0.046, pad=0.02)

        fig.suptitle(f"{sample_id} — {name}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_dir / f"{sample_id}_{_safe(name)}_4panel.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    _write_segmentation_zoom(sample_id, reader, tile_xy, attrs, sc, out_dir, args)
    log.info("[%s] wrote 4-panel figures + segmentation QC -> %s", sample_id, out_dir)


def _write_segmentation_zoom(sample_id, reader, tile_xy, attrs, sc, out_dir, args):
    """Full-resolution crop with the nucleus outlines drawn on it.

    At thumbnail scale a nucleus is sub-pixel, so the 4-panel can only show density.
    This is the figure that actually lets you judge whether segmentation is any good.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cells_xy = sc["cell_xy"]
    if len(cells_xy) == 0:
        return
    read_size = int(attrs["tile_size"])
    zoom = int(args.seg_zoom_size)

    # Centre the crop on the densest tile, so the QC shows a real cell-rich region.
    counts = {}
    for cx, cy in cells_xy:
        key = (int(cx // read_size), int(cy // read_size))
        counts[key] = counts.get(key, 0) + 1
    (bx, by), _ = max(counts.items(), key=lambda kv: kv[1])
    cx0 = int(np.clip(bx * read_size + read_size // 2 - zoom // 2, 0,
                      max(0, reader.width - zoom)))
    cy0 = int(np.clip(by * read_size + read_size // 2 - zoom // 2, 0,
                      max(0, reader.height - zoom)))

    crop = np.asarray(reader.read_region(cx0, cy0, zoom))
    sel = ((cells_xy[:, 0] >= cx0) & (cells_xy[:, 0] < cx0 + zoom) &
           (cells_xy[:, 1] >= cy0) & (cells_xy[:, 1] < cy0 + zoom))
    polys = [sc["cell_poly"][i] - np.array([cx0, cy0]) for i in np.nonzero(sel)[0]]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(crop)
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#00ff88",
                                         linewidths=0.8))
    ax.set_title(f"{sample_id} — StarDist nuclei, full resolution "
                 f"({len(polys)} cells in a {zoom}x{zoom}px crop at {cx0},{cy0})",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_id}_segmentation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_cell_table(sample_id, names, sc, out_dir):
    """Write the per-nucleus table: CSV always, GeoParquet with polygons if possible."""
    cells_xy = sc["cell_xy"]
    # None under --no-save-attention -- the block is then omitted entirely rather
    # than written as a column of NaNs.
    attention = None if sc["cell_attention"] is None else np.asarray(sc["cell_attention"])
    df = pd.DataFrame(sc["cell_score"], columns=names)
    if attention is None:
        pass
    elif attention.ndim == 2:
        # Per-gene attention: one weight per nucleus per target. Appended rather
        # than inserted, so the identifier columns keep their positions.
        for j, name in enumerate(names):
            df[f"attention_{name}"] = attention[:, j]
    else:
        df.insert(0, "attention", attention)
    df.insert(0, "area", sc["cell_area"])
    df.insert(0, "y", cells_xy[:, 1])
    df.insert(0, "x", cells_xy[:, 0])
    df.insert(0, "cell_id", np.arange(len(df)))
    df.to_csv(out_dir / "cells.csv", index=False)

    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        geom = [Polygon(p) for p in sc["cell_poly"]]
        gdf = gpd.GeoDataFrame(df, geometry=geom)
        gdf.to_parquet(out_dir / "cells.parquet")
    except Exception as exc:
        log.warning("[%s] cells.parquet not written (%s); cells.csv is complete",
                    sample_id, exc)
    log.info("[%s] wrote %d nuclei -> cells.csv", sample_id, len(df))


# ------------------------------------------------------------------------------ main


def _resolve_slides(slide_args) -> list:
    slides = []
    for s in slide_args:
        p = Path(s)
        if p.is_dir():
            slides.extend(sorted(f for f in p.iterdir()
                                 if f.suffix.lower() in IMAGE_EXTS))
        elif p.exists():
            slides.append(p)
        else:
            raise FileNotFoundError(f"No such slide or directory: {p}")
    if not slides:
        raise RuntimeError("No slide images found")
    return slides


def build_parser():
    p = argparse.ArgumentParser(
        description="Predict gene / gene-module scores from an H&E slide with a trained "
                    "PathMIL checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--slide", required=True, action="append",
                   help="Slide image, or a directory of them. Repeatable.")
    p.add_argument("--checkpoint", required=True, help="Trained PathMIL checkpoint (.pth)")
    p.add_argument("--out", required=True, help="Output directory")

    p.add_argument("--target-mpp", type=float, default=None,
                   help="Microns-per-pixel the model was TRAINED at. Tiles are read at "
                        "the matching physical size and resized to --tile-size. Defaults "
                        "to the value recorded in the checkpoint; pass 0 to force native "
                        "resolution when the checkpoint carries one.")
    p.add_argument("--tile-size", type=int, default=224, help="Model input tile size (px)")
    p.add_argument("--stride", type=int, default=224,
                   help="Stride between tiles, in model-tile px (scaled with --target-mpp)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-tissue", type=float, default=0.10,
                   help="Drop tiles with less than this fraction of non-white pixels")
    p.add_argument("--white-threshold", type=int, default=220,
                   help="Pixels below this intensity count as tissue")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="Stop after this many tissue tiles (useful for a quick test)")

    p.add_argument("--no-heatmaps", action="store_true", help="Skip the PNG heatmaps")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Named targets to render (default: the --n-top most variable)")
    p.add_argument("--n-top", type=int, default=6)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--thumbnail-max-dim", type=int, default=2500)

    g = p.add_argument_group("single-cell prediction (--sc_pred)")
    g.add_argument("--sc_pred", action="store_true",
                   help="Also segment nuclei (StarDist) and predict at instance (16x16 "
                        "tokens per tile) and single-cell level, with 4-panel figures.")
    g.add_argument("--seg-margin", type=int, default=32,
                   help="Native px of context read around each tile for segmentation. A "
                        "nucleus is kept by the tile containing its centroid, so cells on "
                        "tile seams are counted exactly once.")
    g.add_argument("--min-cell-area", type=float, default=30,
                   help="Drop nuclei smaller than this many native px^2")
    g.add_argument("--dedup-radius", type=float, default=3.0,
                   help="Collapse nuclei whose centroids are within this many px: the two "
                        "tiles sharing a seam can each place a boundary nucleus in their "
                        "own core. 0 disables.")
    g.add_argument("--seg-prob-thresh", type=float, default=0.2)
    g.add_argument("--seg-nms-thresh", type=float, default=0.6)
    g.add_argument("--seg-zoom-size", type=int, default=1024,
                   help="Side length of the full-res segmentation QC crop")
    g.add_argument("--stardist-model-dir", default=None,
                   help="Local dir holding StarDist2D/2D_versatile_he (offline use). "
                        "Default: download the pretrained model on first run.")
    g.add_argument("--max-instance-gb", type=float, default=8.0,
                   help="Refuse to hold the (n_tiles, 256, n_targets) instance array if "
                        "it would exceed this; pass --targets to restrict instead.")
    g.add_argument("--no-save-attention", action="store_true",
                   help="Drop the attention outputs: no instance_attention.npy, and no "
                        "attention column(s) in cells.csv/.parquet. Under a per-gene "
                        "checkpoint the attention array is the same size as the instance "
                        "store, so this roughly halves both peak memory and output size; "
                        "under shared attention it only saves a small file. Instance and "
                        "cell predictions are unchanged.")

    p.add_argument("--save-embeddings", action="store_true",
                   help="Also write the raw (n_tiles, 256, 1280) Virchow2 tokens to H5. "
                        "Large: tens of GB for a whole-slide image.")
    p.add_argument("--foundation-model", default="hf-hub:paige-ai/Virchow2")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--force", action="store_true", help="Recompute slides already done")
    return p


def main(argv=None):
    import torch

    args = build_parser().parse_args(argv)
    slides = _resolve_slides(args.slide)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    names = list(ckpt.get("target_names", []))
    if not names:
        names = [f"target_{i}" for i in range(ckpt["config"]["num_outputs"])]
    log.info("Checkpoint %s: %s, %d outputs, running on %s",
             Path(args.checkpoint).name, ckpt.get("target_type", "?"), len(names), device)

    # The field of view the model was trained at. Reading it from the checkpoint
    # is the whole point of recording it: showing a model a different physical
    # area than it was trained on degrades predictions silently, and requiring the
    # user to remember the number is how that happens.
    if args.target_mpp is None and ckpt.get("target_mpp"):
        args.target_mpp = float(ckpt["target_mpp"])
        log.info("Using --target-mpp %.4f from the checkpoint (%.1f um per tile); "
                 "pass --target-mpp 0 to tile at native resolution instead",
                 args.target_mpp, ckpt.get("fov_um", args.target_mpp * args.tile_size))
    elif args.target_mpp is None:
        log.warning("Neither --target-mpp nor a checkpoint target_mpp: tiling at the "
                    "slide's native resolution. If this model was trained at a "
                    "different magnification its field of view will be wrong.")
    if args.sc_pred and ckpt["config"].get("attention_mode", "shared") == "per_gene":
        # The attention tensor gains a target axis, so --sc_pred's attention
        # outputs change shape (and the memory it needs roughly doubles).
        if args.no_save_attention:
            log.info(
                "Per-gene attention checkpoint with --no-save-attention: the attention "
                "is not accumulated, so no instance_attention.npy is written, cells.csv "
                "has no attention columns, and the --sc_pred stores need about half the "
                "memory they otherwise would.")
        else:
            log.info(
                "Per-gene attention checkpoint: --sc_pred writes instance_attention.npy "
                "as (n_tiles, 256, n_targets) and cells.csv gets one attention_<target> "
                "column per scored target, not a single `attention` column. Attention is "
                "narrowed by --targets alongside the instance scores, so it needs about "
                "as much memory again as the instance store. Pass --no-save-attention to "
                "drop it.")
    log.info("%d slide(s) to process", len(slides))

    # Which targets get instance / single-cell scores. With --targets we can slice the
    # instance head down to just those columns as we go; otherwise we keep them all so
    # the top-N most variable can be chosen once the bag predictions are in.
    seg_model = inst_cols = inst_names = None
    if args.sc_pred:
        if args.targets:
            inst_names = [t for t in args.targets if t in names]
            if not inst_names:
                raise SystemExit(f"None of --targets are in this checkpoint: {names[:5]}...")
        else:
            inst_names = list(names)
        inst_cols = [names.index(t) for t in inst_names]
        log.info("Loading StarDist 2D_versatile_he (CPU; the GPU is kept for Virchow2)")
        seg_model = _load_stardist(args.stardist_model_dir)

    # Virchow2 is a large ViT; load it once and reuse it across every slide.
    virchow, transform = _load_virchow(args.foundation_model, device)

    for slide_path in slides:
        sample_id = _safe(slide_path.stem)
        out_dir = out_root / sample_id
        if (out_dir / "bag_predictions.npy").exists() and not args.force:
            log.info("[%s] already done, skipping (use --force to redo)", sample_id)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        emb_h5 = (out_dir / f"{sample_id}_patch_embeddings.h5") if args.save_embeddings else None
        reader = SlideReader(slide_path)
        try:
            tile_xy, bag, attrs, sc = predict_slide(
                slide_path, reader, virchow, transform, model, args, device,
                inst_cols=inst_cols, seg_model=seg_model, emb_h5_path=emb_h5)

            np.save(out_dir / "bag_predictions.npy", bag)
            np.save(out_dir / "tile_xy.npy", tile_xy)
            df = pd.DataFrame(bag, columns=names)
            df.insert(0, "tile_y", tile_xy[:, 1])
            df.insert(0, "tile_x", tile_xy[:, 0])
            df.to_csv(out_dir / "predictions.csv", index=False)

            if sc is not None:
                np.save(out_dir / "instance_predictions.npy", sc["instance"])
                if sc["attention"] is not None:
                    np.save(out_dir / "instance_attention.npy", sc["attention"])
                pd.DataFrame({"index": range(len(inst_names)), "target": inst_names}).to_csv(
                    out_dir / "instance_target_names.csv", index=False)
                write_cell_table(sample_id, inst_names, sc, out_dir)

            if not args.no_heatmaps:
                write_heatmaps(sample_id, reader, tile_xy, bag, names, attrs, out_dir, args)
                if sc is not None:
                    write_sc_figures(sample_id, reader, tile_xy, bag, names, inst_names,
                                     attrs, sc, out_dir, args)
        finally:
            reader.close()
        log.info("[%s] done -> %s", sample_id, out_dir)

    log.info("All slides finished.")


if __name__ == "__main__":
    main()

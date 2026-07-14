#!/usr/bin/env python
"""Predict gene / gene-module scores directly from an H&E slide.

One script, three stages, no intermediate files:

    slide -> tissue tiles -> Virchow2 patch tokens -> MIL regressor -> per-tile scores

Each tissue tile is one MIL bag: Virchow2 returns 256 spatial tokens per 224px tile,
which is exactly the `(n_bags, 256, 1280)` input the SpaMIL regressor was trained on.
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

Example
-------
    python scripts/predict_he.py \
        --slide tumour.svs \
        --checkpoint model_checkpoint.pth \
        --out results/ \
        --target-mpp 0.2535
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The script lives in scripts/; make the sibling package importable when run from a
# source checkout that has not been pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spamil.model import load_checkpoint  # noqa: E402
from spamil.utils import get_logger  # noqa: E402

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


def predict_slide(slide_path: Path, reader: SlideReader, virchow, transform, model, args,
                  device, emb_h5_path: Path | None = None):
    """Stream one slide through Virchow2 -> MIL; return (tile_xy, bag_preds, attrs).

    Only the batch currently in flight is held in memory. When `emb_h5_path` is given
    the raw tokens are appended to a resizable H5 dataset as they are computed, so even
    --save-embeddings never accumulates the whole slide in RAM.
    """
    import torch
    from PIL import Image

    log.info("[%s] %d x %d px, mpp=%s", slide_path.name, reader.width, reader.height,
             f"{reader.mpp:.4f}" if reader.mpp else "unknown")

    read_size, native_stride = _fov_geometry(reader, args.tile_size, args.stride,
                                             args.target_mpp)
    coords = _tile_coords(reader.width, reader.height, read_size, native_stride)
    log.info("[%s] %d candidate tiles", slide_path.name, len(coords))

    kept: list = []
    preds: list = []
    batch: list = []
    batch_xy: list = []

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
            bag = model(tokens.float())                   # (B, n_outputs)
        preds.append(bag.cpu().numpy())
        if emb_ds is not None:
            arr = tokens.cpu().float().numpy()
            start = emb_ds.shape[0]
            emb_ds.resize((start + arr.shape[0], 256, 1280))
            emb_ds[start:start + arr.shape[0]] = arr
        kept.extend(batch_xy)
        batch.clear()
        batch_xy.clear()

    try:
        n_seen = 0
        for x, y in coords:
            img = reader.read_region(x, y, read_size)
            if read_size != args.tile_size:
                img = img.resize((args.tile_size, args.tile_size), Image.LANCZOS)
            if _tissue_fraction(img, args.white_threshold) < args.min_tissue:
                continue
            batch.append(transform(img).unsqueeze(0))
            batch_xy.append((x, y))
            n_seen += 1
            if len(batch) >= args.batch_size:
                flush()
                log.info("[%s] %d tiles embedded + predicted", slide_path.name, len(kept))
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
            "slide_mpp": reader.mpp or 0.0,
            "target_mpp": args.target_mpp or 0.0,
        }
        log.info("[%s] %d tissue tiles x %d outputs", slide_path.name, *bag_preds.shape)

        if emb_file is not None:
            emb_file.create_dataset("tile_xy", data=tile_xy)
            for k, v in attrs.items():
                emb_file.attrs[k] = v
            emb_file.attrs["n_spots"] = len(tile_xy)
            emb_file.attrs["sample_id"] = _safe(slide_path.stem)
            emb_file.attrs["source_image"] = str(slide_path)
        return tile_xy, bag_preds, attrs
    finally:
        if emb_file is not None:
            emb_file.close()


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

    if args.targets:
        idxs = [names.index(t) for t in args.targets if t in names]
        missing = [t for t in args.targets if t not in names]
        if missing:
            log.warning("[%s] not in this checkpoint: %s", sample_id, ", ".join(missing))
    else:
        idxs = list(np.argsort(np.nanvar(bag, axis=0))[::-1][:args.n_top])

    for i in idxs:
        vmin, vmax = _vlim(bag[:, i])
        fig, ax = plt.subplots(figsize=(9, 8))
        im = _overlay(ax, thumb, _score_grid(tile_xy, bag[:, i], step, W, H),
                      W, H, args.cmap, vmin, vmax, f"{sample_id} — {names[i]}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="predicted score")
        fig.savefig(out_dir / f"{sample_id}_{_safe(names[i])}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    log.info("[%s] wrote %d heatmap PNGs -> %s", sample_id, len(idxs) + 1, out_dir)


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
                    "SpaMIL checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--slide", required=True, action="append",
                   help="Slide image, or a directory of them. Repeatable.")
    p.add_argument("--checkpoint", required=True, help="Trained SpaMIL checkpoint (.pth)")
    p.add_argument("--out", required=True, help="Output directory")

    p.add_argument("--target-mpp", type=float, default=None,
                   help="Microns-per-pixel the model was TRAINED at. Tiles are read at "
                        "the matching physical size and resized to --tile-size. Omit to "
                        "tile at the slide's native resolution.")
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
    log.info("%d slide(s) to process", len(slides))

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
            tile_xy, bag, attrs = predict_slide(slide_path, reader, virchow, transform,
                                                model, args, device, emb_h5_path=emb_h5)

            np.save(out_dir / "bag_predictions.npy", bag)
            np.save(out_dir / "tile_xy.npy", tile_xy)
            df = pd.DataFrame(bag, columns=names)
            df.insert(0, "tile_y", tile_xy[:, 1])
            df.insert(0, "tile_x", tile_xy[:, 0])
            df.to_csv(out_dir / "predictions.csv", index=False)

            if not args.no_heatmaps:
                write_heatmaps(sample_id, reader, tile_xy, bag, names, attrs, out_dir, args)
        finally:
            reader.close()
        log.info("[%s] done -> %s", sample_id, out_dir)

    log.info("All slides finished.")


if __name__ == "__main__":
    main()

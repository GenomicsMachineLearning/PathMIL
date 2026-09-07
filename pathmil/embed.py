"""Step 1 (`pathmil preprocess`): Visium + H&E -> Virchow2 patch embeddings.

Ported from scripts/prepare_training_dataset.py (`process_sample`, `add_spot_patches`,
`stardist`, `zip_zarr_folder`). Produces, per sample, under work_dir/embeddings/:
  - <lib>_patch_embeddings.h5        spot-based bags ('embeddings' (n_spots, 256, 1280)
                                     + attrs['n_spots']); feeds build-targets/train/PCC
  - <lib>_image_patch_embeddings.h5  no-gap sliding-window grid over the tissue
                                     ('embeddings' (n_patches, 256, 1280)); feeds the
                                     whole-tissue visualization
  - <lib>.zarr.zip                   SpatialData archive (incl. the matching
                                     `image_patches` grid), used by plotting

Requires the heavy `embed` extras (sopa, spatialdata, timm, stardist, dask, ...).
"""

from __future__ import annotations

import os
import sys
import shutil
import warnings
import zipfile
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from pathmil.config import cget
from pathmil import io as sio
from pathmil.scale import resolve_patch_geometry
from pathmil.utils import get_logger, hf_login_if_enabled

log = get_logger()


class _SuppressPrintsAndWarnings:
    def __enter__(self):
        self._stdout = sys.stdout
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull
        self._filters = warnings.filters[:]
        warnings.simplefilter("ignore")

    def __exit__(self, *args):
        sys.stdout = self._stdout
        try:
            self._devnull.close()
        except Exception:
            pass
        warnings.filters = self._filters


def _tf_cpu_only():
    """Hide the GPU from TensorFlow so StarDist runs on CPU.

    sopa fans StarDist out over several dask workers; if each TF process claims the
    GPU they collectively OOM it. The GPU is reserved for the Virchow2 (torch) pass,
    so StarDist (a small model, not on the embedding path) is pinned to CPU. Must be
    called right after importing TF, before any GPU is initialised.
    """
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _stardist_patch(model_type, prob_thresh, nms_thresh, basedir, **kw) -> Callable:
    _tf_cpu_only()
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    def _(patch, model_type, prob_thresh, nms_thresh, **kw):
        with _SuppressPrintsAndWarnings():
            _tf_cpu_only()  # re-applied inside each (freshly-imported) dask worker
            if basedir:
                model = StarDist2D(None, name="2D_versatile_he", basedir=basedir)
            else:
                model = StarDist2D.from_pretrained("2D_versatile_he")
            patch = normalize(patch.transpose(1, 2, 0), clip=True)
            mask, _ = model.predict_instances(
                patch, prob_thresh=prob_thresh, nms_thresh=nms_thresh, **kw)
            return mask

    return partial(_, model_type=model_type, prob_thresh=prob_thresh,
                   nms_thresh=nms_thresh, **kw)


def _stardist(sdata, image_key, basedir, min_area=0, prob_thresh=0.2, nms_thresh=0.6,
              key_added="stardist_HE"):
    from sopa.segmentation.methods._custom import custom_staining_based
    method = _stardist_patch("2D_versatile_he", prob_thresh, nms_thresh, basedir)
    custom_staining_based(
        sdata, method, channels=None, image_key=image_key, min_area=min_area,
        delete_cache=True, recover=False, clip_limit=0, clahe_kernel_size=None,
        gaussian_sigma=0, cache_dir_name=key_added, key_added=key_added)


def _add_spot_patches(sdata, library_id, size=224):
    import shapely.geometry
    df = sdata[f"{library_id}"].copy()
    df["x"] = df["geometry"].apply(lambda pt: pt.x)
    df["y"] = df["geometry"].apply(lambda pt: pt.y)

    def to_square(point, size=size):
        x, y = point.x, point.y
        half = size / 2
        corners = [(x - half, y - half), (x + half, y - half), (x + half, y + half),
                   (x - half, y + half), (x - half, y - half)]
        return shapely.geometry.Polygon(corners)

    df["geometry"] = df["geometry"].apply(lambda pt: to_square(pt, size=size))
    df["bboxes"] = df["geometry"].apply(lambda poly: list(poly.bounds))
    sdata["spot_patches"] = df


def _save_qc_overlay(sdata, image_key, shapes_key, title, out_file) -> None:
    """Best-effort QC figure: outline `shapes_key` over the downsampled H&E and save it.

    Reuses the spatialdata-plot `.pl` accessor pattern from pathmil/plot.py. Never raises:
    QC rendering must not abort preprocessing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import spatialdata_plot  # noqa: F401 -- registers the .pl accessor on SpatialData

        fig, ax = plt.subplots(figsize=(10, 10))
        (sdata.pl.render_images(image_key, scale="scale3")
              .pl.render_shapes(shapes_key, fill_alpha=0, outline_alpha=1, outline_width=0.5)
              .pl.show(ax=ax, title=title))
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("[%s] QC plot -> %s", title, out_file)
    except Exception as exc:
        log.warning("QC overlay '%s' (shapes=%s) failed: %s", title, shapes_key, exc)


def _zip_zarr_folder(tmp_path: Path, library_id: str) -> Optional[Path]:
    zarr_folder = tmp_path / f"{library_id}.zarr"
    if not zarr_folder.is_dir():
        log.warning("Zarr folder not found: %s", zarr_folder)
        return None
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        zip_file = tmp_path / f"{zarr_folder.name}.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in zarr_folder.rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(zarr_folder))
        return zip_file
    finally:
        os.chdir(original_cwd)


def process_sample(library_id: str, cfg: dict, paths: dict, force: bool = False) -> None:
    """Generate patch embeddings + zarr archive for one sample."""
    import torch
    import h5py
    import sopa
    import spatialdata as sd
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked
    from PIL import Image
    import dask
    from dask import delayed

    Image.MAX_IMAGE_PIXELS = None

    emb_dir = Path(paths["embeddings"])
    out_h5 = emb_dir / f"{library_id}_patch_embeddings.h5"
    out_img_h5 = emb_dir / f"{library_id}_image_patch_embeddings.h5"
    out_zip = emb_dir / f"{library_id}.zarr.zip"
    if out_h5.exists() and out_img_h5.exists() and out_zip.exists() and not force:
        log.info("[%s] embeddings already exist, skipping (use --force to redo)", library_id)
        return

    tmp_path = Path(paths["tmp_dir"])
    tmp_path.mkdir(parents=True, exist_ok=True)
    emb_batch = int(cget(cfg, "embed.batch_size", 32))
    stardist_dir = cget(cfg, "embed.stardist_model_dir")
    min_area = int(cget(cfg, "embed.stardist_min_area", 30))
    cell_seg_width = int(cget(cfg, "embed.cell_seg_patch_width", 2000))
    cell_seg_overlap = int(cget(cfg, "embed.cell_seg_patch_overlap", 50))
    save_debug_plots = bool(cget(cfg, "embed.save_debug_plots", True))
    plots_dir = Path(paths["plots"]) / library_id if save_debug_plots else None

    # Physical patch scale. With `embed.target_mpp` set, patches are read at
    # whatever native size covers `patch_size * target_mpp` micrometres and are
    # resized to patch_size by the model transform, so every slide hands the
    # foundation model the same quantity of tissue regardless of the scanner.
    # Unset, this is exactly the historical native-pixel path.
    geom = resolve_patch_geometry(library_id, cfg)
    if geom["target_mpp"] is None:
        log.info("[%s] patch scale: native pixels (embed.target_mpp unset), spot %d px, "
                 "image grid %d px", library_id, geom["spot_read_px"], geom["image_read_px"])
    else:
        log.info("[%s] patch scale: slide %.4f um/px -> target %.4f um/px "
                 "(%.1f um FOV); spot read %d px, image grid %d px",
                 library_id, geom["slide_mpp"], geom["target_mpp"], geom["fov_um"],
                 geom["spot_read_px"], geom["image_read_px"])

    sopa.settings.parallelization_backend = "dask"
    sopa.settings.dask_client_kwargs = {
        "n_workers": int(cget(cfg, "embed.n_workers", 6)),
        "threads_per_worker": int(cget(cfg, "embed.threads_per_worker", 1)),
        "memory_limit": cget(cfg, "embed.memory_limit", "40GB"),
    }

    log.info("[%s] loading SpatialData (Visium + H&E)", library_id)
    sdata = sio.load_spatialdata(library_id, cfg)
    image_key = f"{library_id}_full_image"

    # Grid width comes from `geom` (target_mpp-aware); only the overlap is read here.
    img_patch_overlap = int(cget(cfg, "embed.image_patch_overlap", 0))

    log.info("[%s] tissue segmentation", library_id)
    sopa.segmentation.tissue(sdata, image_key=image_key, expand_radius_ratio=0.01)
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_qc_overlay(sdata, image_key, "region_of_interest",
                         f"{library_id} - tissue mask",
                         plots_dir / f"{library_id}_qc_tissue_mask.png")

    # Cell-segmentation grid: large independent tiles for StarDist. `custom_staining_based`
    # hard-codes reading patches from sdata.shapes["image_patches"], so we fill that key with
    # the cell-seg grid now, then overwrite it with the embedding grid after segmentation.
    log.info("[%s] cell-seg grid (width=%d, overlap=%d)",
             library_id, cell_seg_width, cell_seg_overlap)
    sopa.make_image_patches(sdata, patch_width=cell_seg_width,
                            patch_overlap=cell_seg_overlap, image_key=image_key)

    sdata.write(tmp_path / f"{library_id}_temp.zarr", overwrite=True)
    sdata = sd.read_zarr(tmp_path / f"{library_id}_temp.zarr")

    log.info("[%s] stardist segmentation", library_id)
    _stardist(sdata, image_key=image_key, basedir=stardist_dir, min_area=min_area)

    # Embedding grid: no-gap sliding-window tiles for the visualization track. Overwrites the
    # cell-seg grid in sdata.shapes["image_patches"] (delete_cache above clears only the
    # on-disk segmentation cache, not these shapes).
    log.info("[%s] image patches (no-gap grid: width=%d, overlap=%d)",
             library_id, geom["image_read_px"], img_patch_overlap)
    sopa.make_image_patches(sdata, patch_width=geom["image_read_px"],
                            patch_overlap=img_patch_overlap, image_key=image_key)
    if plots_dir is not None:
        _save_qc_overlay(sdata, image_key, "image_patches",
                         f"{library_id} - tiles (width={geom['image_read_px']})",
                         plots_dir / f"{library_id}_qc_tiles.png")

    log.info("[%s] spot patches (read size=%d px)", library_id, geom["spot_read_px"])
    _add_spot_patches(sdata, library_id, size=geom["spot_read_px"])

    log.info("[%s] loading foundation model %s", library_id, cget(cfg, "embed.foundation_model"))
    model = timm.create_model(
        cget(cfg, "embed.foundation_model", "hf-hub:paige-ai/Virchow2"),
        pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval()
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    image = next(iter(sdata[image_key]["scale0"].values()))

    def extract_and_transform(row, read_px):
        """Crop `read_px` native pixels around a shape and resize to the model input.

        Under `embed.target_mpp` the crop is larger or smaller than the model's
        patch size; `transforms` resizes it, which is what puts every slide on the
        same physical scale. Padding is measured against `read_px`, so a patch
        clipped at the slide edge is padded rather than silently short.
        """
        box = row["bboxes"]
        patch = image[:, slice(int(box[1]), int(box[3])), slice(int(box[0]), int(box[2]))]
        pad_x = max(0, read_px - patch.shape[1])
        pad_y = max(0, read_px - patch.shape[2])
        patch = np.pad(patch, ((0, 0), (0, pad_x), (0, pad_y)))
        patch = patch.transpose(1, 2, 0)
        return transforms(Image.fromarray(patch)).unsqueeze(0)

    def embed_shapes(shapes_key, out_h5, read_px):
        """Extract patches for `sdata[shapes_key]`, embed them, and write `out_h5`."""
        log.info("[%s] extracting/transforming patches for '%s' (read %d px)",
                 library_id, shapes_key, read_px)
        tasks = [delayed(extract_and_transform)(row, read_px)
                 for _, row in sdata[shapes_key].iterrows()]
        all_tensors = dask.compute(*tasks)
        batch = torch.cat(all_tensors, dim=0)  # (N, 3, P, P)

        n_patches = batch.shape[0]
        token_shape = (256, 1280)
        log.info("[%s] computing embeddings for %d '%s' patches",
                 library_id, n_patches, shapes_key)
        with h5py.File(out_h5, "w") as f:
            ds = f.create_dataset("embeddings", shape=(n_patches, *token_shape),
                                  dtype="float32",
                                  chunks=(min(emb_batch, n_patches), *token_shape))

            def process_and_save(i):
                chunk = batch[i:i + emb_batch]
                with torch.no_grad():
                    out = model(chunk)
                    ds[i:i + chunk.shape[0]] = out[:, 5:].cpu().float().numpy()

            dask.compute([delayed(process_and_save)(i)
                          for i in range(0, n_patches, emb_batch)],
                         scheduler="threads")
            f.attrs["n_spots"] = n_patches
            # The scale these embeddings were cut at. Without it nothing
            # downstream can say what physical field of view the model was shown,
            # and `scripts/predict_he.py --target-mpp` has to be remembered by hand.
            f.attrs["patch_size"] = geom["patch_size"]
            f.attrs["read_px"] = read_px
            if geom["target_mpp"] is not None:
                f.attrs["target_mpp"] = geom["target_mpp"]
                f.attrs["slide_mpp"] = geom["slide_mpp"]
                f.attrs["fov_um"] = geom["fov_um"]

    # Spot-based bags (PCC / training) and the no-gap sliding-window grid (visualization).
    tmp_h5 = tmp_path / f"{library_id}_patch_embeddings.h5"
    tmp_img_h5 = tmp_path / f"{library_id}_image_patch_embeddings.h5"
    embed_shapes("spot_patches", tmp_h5, geom["spot_read_px"])
    embed_shapes("image_patches", tmp_img_h5, geom["image_read_px"])

    log.info("[%s] writing zarr archive", library_id)
    sdata.write(tmp_path / f"{library_id}.zarr", overwrite=True)
    zip_path = _zip_zarr_folder(tmp_path, library_id)

    emb_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_h5), str(out_h5))
    shutil.move(str(tmp_img_h5), str(out_img_h5))
    if zip_path:
        shutil.move(str(zip_path), str(out_zip))
        log.info("[%s] done -> %s (+ %s + zarr.zip)",
                 library_id, out_h5.name, out_img_h5.name)
    else:
        log.warning("[%s] embeddings written but zarr archive failed", library_id)


def run_preprocess(cfg: dict, paths: dict, samples=None, index=None, force=False) -> None:
    """Driver for `pathmil preprocess`."""
    hf_login_if_enabled(cfg)
    all_samples = samples if samples else sio.discover_samples(cfg)
    if not all_samples:
        raise RuntimeError("No samples discovered; check dataset.data_dir / dataset.layout")

    if index is not None:
        # SLURM-array compatibility: 0-based index into the discovered list.
        if index < 0 or index >= len(all_samples):
            raise IndexError(f"--index {index} out of range (0-{len(all_samples) - 1})")
        all_samples = [all_samples[index]]

    log.info("Preprocessing %d sample(s): %s", len(all_samples), ", ".join(all_samples))
    for lib in all_samples:
        try:
            process_sample(lib, cfg, paths, force=force)
        except Exception:  # keep the array job moving; surface the traceback
            log.exception("[%s] preprocessing FAILED", lib)
            raise

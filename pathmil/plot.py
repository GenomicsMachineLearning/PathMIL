"""Step 5 (`pathmil plot`): per-gene 4-panel prediction maps on the H&E.

For each selected target, render one figure with four panels:

  1. ground truth, spot level     -- real Visium expression on the spots
  2. prediction, spot level       -- bag_predictions on the spots
  3. prediction, instance level   -- instance_predictions as a 16x16 "superpixel"
                                     grid inside each spot's patch
  4. prediction, single-cell      -- instance predictions aggregated onto StarDist cells

This consumes the SpatialData archive `<lib>.zarr.zip` written by the embed step
(spots, full image, `spot_patches` + the no-gap `image_patches` grid with bboxes,
`stardist_HE` cell shapes, and the expression `table`). It therefore needs the heavy
`embed` extras (spatialdata, spatialdata-plot, geopandas, shapely); those are imported
lazily so importing this module stays cheap.

The 256 instances per patch are the Virchow2 token grid -- a 16x16 spatial grid inside
each patch -- treated here as a grid of "superpixels". Panels 1-2 (ground truth /
spot bag prediction) render onto the spot shapes `sdata["<lib>"]`. Panels 3-4 prefer
the visualization track from `pathmil predict`: `image_instance_predictions.npy` over
the `image_patches` sliding-window grid (whole-tissue coverage, as in
notebooks/05_prediction_plot_test.ipynb). When that track is absent they fall back to
the per-spot `instance_predictions.npy` over `spot_patches`.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathmil.config import cget
from pathmil.utils import get_logger

log = get_logger()

PATCHES_PER_SIDE = 16  # Virchow2 token grid -> 16x16 = 256 instances per spot


def _load_prediction_dir(paths, lib, prediction_dir):
    """Locate the directory holding bag_predictions.npy for a sample."""
    if prediction_dir:
        d = Path(prediction_dir) / lib
        if (d / "bag_predictions.npy").exists():
            return d
        if (Path(prediction_dir) / "bag_predictions.npy").exists():
            return Path(prediction_dir)
    d = Path(paths["predictions"]) / lib
    if (d / "bag_predictions.npy").exists():
        return d
    raise FileNotFoundError(
        f"No bag_predictions.npy for {lib}; run `pathmil predict` or pass --prediction-dir")


def _target_names(pred_dir: Path, n: int) -> list:
    f = pred_dir / "target_names.csv"
    if f.exists():
        return pd.read_csv(f)["target"].tolist()
    f2 = pred_dir / "target_level_pcc.csv"
    if f2.exists():
        return pd.read_csv(f2)["target"].tolist()
    return [f"output_{i}" for i in range(n)]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _select_targets(names, bag, targets, n_top, lib):
    """Return the list of (idx, name) targets to render for a sample."""
    if targets:
        idxs = [names.index(t) for t in targets if t in names]
        missing = [t for t in targets if t not in names]
        if missing:
            log.warning("[%s] targets not found: %s", lib, ", ".join(missing))
    else:
        idxs = list(np.argsort(bag.var(axis=0))[::-1][:n_top])  # most spatially variable
    return [(int(j), names[j]) for j in idxs]


def _read_zarr(lib, cfg, paths):
    """Extract `<lib>.zarr.zip` into tmp_dir and read it; return the SpatialData (or None)."""
    import spatialdata as sd

    zip_path = Path(paths["embeddings"]) / f"{lib}.zarr.zip"
    if not zip_path.exists():
        log.warning("[%s] no zarr archive at %s; skipping", lib, zip_path)
        return None
    tmp_root = Path(paths["tmp_dir"])
    tmp_root.mkdir(parents=True, exist_ok=True)
    zarr_dir = tmp_root / f"{lib}.zarr"
    if not zarr_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zarr_dir)
    return sd.read_zarr(zarr_dir)


def _normalized_table(sdata):
    """Return a normalized + log1p copy of the spot expression table (panel 1 source)."""
    import scanpy as sc

    table = sdata["table"].copy()
    sc.pp.normalize_total(table)
    sc.pp.log1p(table)
    return table


def _build_superpixels(sdata, lib, inst, idx_names, patches_key="spot_patches"):
    """Build the 16x16 per-patch superpixel shapes + table from instance predictions.

    `inst` is (n_patches, 256, n_out); `idx_names` is the list of (gene_idx, gene_name)
    being rendered. `patches_key` selects the bag geometry: "image_patches" (the no-gap
    sliding-window grid covering the whole tissue) or "spot_patches" (per-spot). Adds
    `superpixel_predictions` (+ `_table`) to `sdata` in place.
    """
    import anndata as ad
    import geopandas as gpd
    from shapely.geometry import Polygon
    from spatialdata.models import ShapesModel, TableModel
    from spatialdata.transformations import get_transformation, set_transformation

    bboxes = sdata[patches_key]["bboxes"].values
    n_spots = len(bboxes)
    n = min(n_spots, inst.shape[0])

    polygons, instance_ids = [], []
    spot_idx_col, sp_idx_col = [], []
    pred_cols = {name: [] for _, name in idx_names}

    for spot_idx in range(n):
        x_min, y_min, x_max, y_max = bboxes[spot_idx]
        w = (x_max - x_min) / PATCHES_PER_SIDE
        h = (y_max - y_min) / PATCHES_PER_SIDE
        for row in range(PATCHES_PER_SIDE):
            for col in range(PATCHES_PER_SIDE):
                sx = x_min + col * w
                sy = y_min + row * h
                polygons.append(Polygon([(sx, sy), (sx + w, sy),
                                         (sx + w, sy + h), (sx, sy + h)]))
                sp_idx = row * PATCHES_PER_SIDE + col
                instance_ids.append(f"spot_{spot_idx}_sp_{row}_{col}")
                spot_idx_col.append(spot_idx)
                sp_idx_col.append(sp_idx)
                for gene_idx, name in idx_names:
                    pred_cols[name].append(inst[spot_idx, sp_idx, gene_idx])

    gdf_data = {"geometry": polygons, "spot_idx": spot_idx_col, "super_pixel_idx": sp_idx_col}
    for _, name in idx_names:
        gdf_data[f"{name}_pred"] = pred_cols[name]
    gdf = gpd.GeoDataFrame(gdf_data, index=instance_ids)

    shapes = ShapesModel.parse(gdf)
    coord_system = lib
    transformation = get_transformation(sdata[patches_key], coord_system)
    set_transformation(shapes, transformation, to_coordinate_system=coord_system)
    sdata["superpixel_predictions"] = shapes

    obs = pd.DataFrame({"spot_idx": spot_idx_col, "super_pixel_idx": sp_idx_col})
    for _, name in idx_names:
        obs[f"{name}_pred"] = pred_cols[name]
    adata = ad.AnnData(obs.astype("float32"))
    adata.obs["instance_id"] = instance_ids
    adata.obs["region"] = "superpixel_predictions"
    adata.obs.index = instance_ids
    adata.uns["spatialdata_attrs"] = {
        "region": "superpixel_predictions",
        "region_key": "region",
        "instance_key": "instance_id",
    }
    sdata["superpixel_predictions_table"] = TableModel.parse(adata)
    return gdf


def _aggregate_to_cells(sdata, gdf_superpixels, idx_names):
    """Aggregate superpixel predictions onto StarDist cells; add a `stardist_HE` table."""
    import anndata as ad
    import geopandas as gpd
    from spatialdata.models import TableModel

    stardist_gdf = sdata["stardist_HE"].copy()
    for _, name in idx_names:
        col = f"{name}_pred"
        joined = gpd.sjoin(gdf_superpixels[["geometry", col]], stardist_gdf,
                           how="right", predicate="intersects")
        if len(joined) > 0:
            mean_pred = joined.groupby(joined.index)[col].mean()
            stardist_gdf[col] = stardist_gdf.index.map(mean_pred)
            fill = stardist_gdf[col].min()
            stardist_gdf[col] = stardist_gdf[col].fillna(0.0 if pd.isna(fill) else fill)
        else:
            stardist_gdf[col] = 0.0
    sdata["stardist_HE"] = stardist_gdf

    cell_ids = list(stardist_gdf.index.astype(str))
    obs = pd.DataFrame({f"{name}_pred": stardist_gdf[f"{name}_pred"].values
                        for _, name in idx_names})
    adata = ad.AnnData(obs.astype("float32"))
    adata.obs["cell_id"] = cell_ids
    adata.obs["region"] = "stardist_HE"
    adata.obs.index = cell_ids
    adata.uns["spatialdata_attrs"] = {
        "region": "stardist_HE",
        "region_key": "region",
        "instance_key": "cell_id",
    }
    sdata["stardist_HE_table"] = TableModel.parse(adata)


def _placeholder(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)


def _panel_figure(sdata, lib, gene, out_dir, cmap):
    """Render and save the 2x2 prediction figure for one gene."""
    img_key = f"{lib}_full_image"
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: ground truth (spot-level expression).
    try:
        sdata.pl.render_images(img_key, scale="scale3").pl.render_shapes(
            lib, color=gene, cmap=cmap, table_name="table",
        ).pl.show(lib, title=f"{gene} - ground truth (spot)", ax=axes[0, 0])
    except Exception as exc:
        log.warning("[%s] %s ground-truth panel failed: %s", lib, gene, exc)
        _placeholder(axes[0, 0], f"{gene}\nground truth\n(not available)")
        axes[0, 0].set_title(f"{gene} - ground truth (spot)")

    # Panel 2: spot-level prediction (bag predictions on the spots).
    try:
        sdata.pl.render_images(img_key, scale="scale3").pl.render_shapes(
            lib, color=f"{gene}_bag_pred", cmap=cmap, table_name="table",
        ).pl.show(lib, title=f"{gene} - prediction (spot)", ax=axes[0, 1])
    except Exception as exc:
        log.warning("[%s] %s spot-prediction panel failed: %s", lib, gene, exc)
        _placeholder(axes[0, 1], f"{gene}\nspot prediction\n(not available)")
        axes[0, 1].set_title(f"{gene} - prediction (spot)")

    # Panel 3: instance-level prediction (superpixels).
    try:
        sdata.pl.render_images(img_key, scale="scale3").pl.render_shapes(
            "superpixel_predictions", color=f"{gene}_pred", cmap=cmap,
            table_name="superpixel_predictions_table", datashader_reduction="max",
        ).pl.show(lib, title=f"{gene} - prediction (instance)", ax=axes[1, 0])
    except Exception as exc:
        log.warning("[%s] %s instance panel failed: %s", lib, gene, exc)
        _placeholder(axes[1, 0], f"{gene}\ninstance prediction\n(not available)")
        axes[1, 0].set_title(f"{gene} - prediction (instance)")

    # Panel 4: single-cell prediction (aggregated onto StarDist cells).
    try:
        sdata.pl.render_images(img_key, scale="scale3").pl.render_shapes(
            "stardist_HE", color=f"{gene}_pred", cmap=cmap,
            table_name="stardist_HE_table", datashader_reduction="max",
        ).pl.show(lib, title=f"{gene} - prediction (single cell)", ax=axes[1, 1])
    except Exception as exc:
        log.warning("[%s] %s single-cell panel failed: %s", lib, gene, exc)
        _placeholder(axes[1, 1], f"{gene}\nsingle-cell prediction\n(not available)")
        axes[1, 1].set_title(f"{gene} - prediction (single cell)")

    fig.tight_layout()
    out_file = out_dir / f"{lib}_{_safe(gene)}_4panel.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _plot_pcc_summary(lib, pcc_csv: Path, out_dir: Path):
    df = pd.read_csv(pcc_csv).dropna(subset=["pcc"])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["pcc"], bins=min(50, max(5, len(df) // 2)), edgecolor="black", alpha=0.7)
    ax.axvline(df["pcc"].mean(), color="red", linestyle="--",
               label=f"mean {df['pcc'].mean():.3f}")
    ax.axvline(df["pcc"].median(), color="green", linestyle="--",
               label=f"median {df['pcc'].median():.3f}")
    ax.set_xlabel("Pearson correlation"); ax.set_ylabel("count")
    ax.set_title(f"{lib} - target PCC distribution"); ax.legend()
    fig.tight_layout()
    out_file = out_dir / f"{lib}_pcc_summary.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _plot_sample(cfg, paths, lib, targets, prediction_dir, n_top, cmap):
    pred_dir = _load_prediction_dir(paths, lib, prediction_dir)
    bag = np.load(pred_dir / "bag_predictions.npy")
    # Panels 3-4 prefer the no-gap sliding-window image grid (whole-tissue coverage);
    # fall back to the per-spot instance grid for older prediction dirs.
    img_inst_path = pred_dir / "image_instance_predictions.npy"
    spot_inst_path = pred_dir / "instance_predictions.npy"
    if img_inst_path.exists():
        inst, patches_key = np.load(img_inst_path), "image_patches"
    elif spot_inst_path.exists():
        inst, patches_key = np.load(spot_inst_path), "spot_patches"
    else:
        inst, patches_key = None, None
    names = _target_names(pred_dir, bag.shape[1])

    idx_names = _select_targets(names, bag, targets, n_top, lib)
    if not idx_names:
        log.warning("[%s] no targets to render", lib)
        return

    sdata = _read_zarr(lib, cfg, paths)
    if sdata is None:
        return

    # Panel 1 source: normalized ground-truth expression in place of the raw table.
    sdata["table"] = _normalized_table(sdata)

    # Panel 2 source: map bag predictions onto the spot table (positional order matches
    # spot_patches / embeddings / prediction order).
    n_spots = sdata["table"].n_obs
    n = min(n_spots, bag.shape[0])
    for gene_idx, name in idx_names:
        col = np.full(n_spots, np.nan, dtype="float32")
        col[:n] = bag[:n, gene_idx]
        sdata["table"].obs[f"{name}_bag_pred"] = col

    # Panels 3-4: build superpixels + single-cell aggregation once for all genes.
    if inst is not None and patches_key in sdata:
        try:
            gdf = _build_superpixels(sdata, lib, inst, idx_names, patches_key=patches_key)
            _aggregate_to_cells(sdata, gdf, idx_names)
            log.info("[%s] superpixels built over '%s' (%d patches)",
                     lib, patches_key, inst.shape[0])
        except Exception:
            log.exception("[%s] failed to build instance/single-cell layers", lib)
    else:
        log.warning("[%s] no usable instance predictions / '%s' shapes; instance & "
                    "single-cell panels will be placeholders", lib, patches_key)

    out_dir = Path(paths["plots"]) / lib
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, name in idx_names:
        f = _panel_figure(sdata, lib, name, out_dir, cmap)
        log.info("[%s] %s -> %s", lib, name, f.name)

    pcc_csv = pred_dir / "target_level_pcc.csv"
    if pcc_csv.exists():
        _plot_pcc_summary(lib, pcc_csv, out_dir)


def run_plot(cfg: dict, paths: dict, samples=None, targets=None, prediction_dir=None,
             n_top=None) -> None:
    import spatialdata_plot  # noqa: F401  (registers the `.pl` accessor)

    cmap = cget(cfg, "plot.cmap", "viridis")
    n_top = int(n_top if n_top is not None else cget(cfg, "plot.n_top", 6))

    if not samples:
        samples = sorted(p.name for p in Path(paths["predictions"]).glob("*") if p.is_dir())
    if not samples:
        raise RuntimeError("No samples to plot; run `pathmil predict` first or pass --samples")

    for lib in samples:
        try:
            _plot_sample(cfg, paths, lib, targets, prediction_dir, n_top, cmap)
        except Exception:
            log.exception("[%s] plotting FAILED", lib)

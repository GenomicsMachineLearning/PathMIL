"""Layout-flexible Visium discovery & loading.

Some datasets use `<data_dir>/<batch>/<lib>/outs/` plus a metadata CSV to resolve
H&E filenames; others use a flat layout `<data_dir>/<lib>/` with H&E in a separate
directory. Rather than auto-probing,
the layout is declared in the dataset YAML (`dataset.layout`: outs | flat) and the
H&E filename is built from `dataset.he_dir` + `dataset.he_pattern`.

Expected structures (documented in the README):

  layout: outs                      layout: flat
  <data_dir>/<lib>/outs/            <data_dir>/<lib>/
      filtered_feature_bc_matrix.h5     filtered_feature_bc_matrix.h5
      spatial/...                       spatial/...
"""

from __future__ import annotations

from pathlib import Path

from spamil.config import cget, expand

H5_NAME = "filtered_feature_bc_matrix.h5"


def visium_dir(library_id: str, cfg: dict) -> Path:
    """Directory that directly contains filtered_feature_bc_matrix.h5 + spatial/."""
    data_dir = Path(expand(cget(cfg, "dataset.data_dir")))
    layout = cget(cfg, "dataset.layout", "flat")
    if layout == "outs":
        return data_dir / library_id / "outs"
    if layout == "flat":
        return data_dir / library_id
    raise ValueError(f"Unknown dataset.layout '{layout}' (expected 'outs' or 'flat')")


def he_path(library_id: str, cfg: dict) -> Path:
    """Resolve the full-resolution H&E image path for a sample."""
    he_dir = cget(cfg, "dataset.he_dir")
    pattern = cget(cfg, "dataset.he_pattern", "{library_id}.tif")
    fname = pattern.format(library_id=library_id)
    if he_dir:
        return Path(expand(he_dir)) / fname
    # No he_dir: the H&E lives inside the sample's own directory.
    return visium_dir(library_id, cfg) / fname


def sample_sheet_ids(cfg: dict) -> list | None:
    """Library ids from `dataset.sample_sheet`, in sheet order, or None if unset.

    The sheet is a tidy CSV with one row per sample; `dataset.sheet_lib_col` names
    the column holding the library id (== sample sub-directory name). Order is
    preserved so SLURM-array `--index` maps stably onto samples.
    """
    sheet = cget(cfg, "dataset.sample_sheet")
    if not sheet:
        return None
    import pandas as pd

    lib_col = cget(cfg, "dataset.sheet_lib_col", "library_id")
    df = pd.read_csv(expand(sheet))
    if lib_col not in df.columns:
        raise KeyError(f"sheet_lib_col '{lib_col}' not in {sheet} (cols: {list(df.columns)})")
    return [str(x) for x in df[lib_col].tolist() if str(x).strip() and str(x) != "nan"]


def discover_samples(cfg: dict) -> list:
    """Return the list of library ids to process.

    Resolution order:
      1. `dataset.samples` is an explicit list -> use it verbatim.
      2. `dataset.sample_sheet` is set -> use its `sheet_lib_col` column (in order),
         filtered to libs whose Visium matrix exists.
      3. else "auto" -> glob the data dir for sub-dirs with a Visium matrix.
    """
    samples = cget(cfg, "dataset.samples", "auto")
    if isinstance(samples, (list, tuple)):
        return list(samples)

    sheet_ids = sample_sheet_ids(cfg)
    if sheet_ids is not None:
        return [lib for lib in sheet_ids
                if (visium_dir(lib, cfg) / H5_NAME).exists()]

    data_dir = Path(expand(cget(cfg, "dataset.data_dir")))
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset.data_dir not found: {data_dir}")

    found = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        # Probe expected matrix location for this library id under the layout.
        if (visium_dir(child.name, cfg) / H5_NAME).exists():
            found.append(child.name)
    return found


def load_spatialdata(library_id: str, cfg: dict):
    """Load a SpatialData object (Visium + full-res H&E). Heavy; needs `embed` extras."""
    import spatialdata_io as sdio

    vdir = visium_dir(library_id, cfg)
    he = he_path(library_id, cfg)
    if not (vdir / H5_NAME).exists():
        raise FileNotFoundError(f"Visium matrix not found: {vdir / H5_NAME}")
    if not he.exists():
        raise FileNotFoundError(f"H&E image not found: {he}")
    return sdio.visium(vdir, dataset_id=library_id, fullres_image_file=he)


def load_expression(library_id: str, cfg: dict):
    """Lightweight load of the spot x gene expression AnnData (no image).

    Used for gene-list (HVG) selection where spot ordering does not matter.
    """
    import scanpy as sc

    h5 = visium_dir(library_id, cfg) / H5_NAME
    if not h5.exists():
        raise FileNotFoundError(f"Visium matrix not found: {h5}")
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()
    return adata


def read_spot_coordinates(library_id: str, cfg: dict):
    """Return (barcodes, xy_pixels) aligned to the filtered matrix spot order.

    Pixel coordinates come from spatial/tissue_positions(.csv), joined by barcode
    so the order matches the embeddings/predictions (which follow the matrix order).
    Lightweight — used only for plotting.
    """
    import h5py
    import numpy as np
    import pandas as pd

    vdir = visium_dir(library_id, cfg)

    # Barcode order as stored in the filtered matrix (== embedding/prediction order).
    with h5py.File(vdir / H5_NAME, "r") as f:
        barcodes = f["matrix/barcodes"][:]
    barcodes = np.array([b.decode() if isinstance(b, bytes) else b for b in barcodes])

    # tissue_positions file (newer: tissue_positions.csv with header; older: list, no header)
    spatial = vdir / "spatial"
    pos_file = None
    for name in ("tissue_positions.csv", "tissue_positions_list.csv"):
        if (spatial / name).exists():
            pos_file = spatial / name
            break
    if pos_file is None:
        raise FileNotFoundError(f"No tissue_positions file under {spatial}")

    if pos_file.name == "tissue_positions.csv":
        pos = pd.read_csv(pos_file)
    else:
        pos = pd.read_csv(pos_file, header=None, names=[
            "barcode", "in_tissue", "array_row", "array_col",
            "pxl_row_in_fullres", "pxl_col_in_fullres"])
    pos = pos.set_index("barcode")

    keep = [b for b in barcodes if b in pos.index]
    pos = pos.loc[keep]
    xy = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    return np.array(keep), xy

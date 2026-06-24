"""Step 2 (`spamil build-targets`): build combined embeddings+target H5 files.

Ported from notebooks 03_train_MIL / 03_02_train_MIL_ms. For each sample it reads
the lightweight expression matrix (spot order == embedding order, both follow the
filtered matrix barcode order), normalises (total + log1p), derives the target
(top-N HVG genes, or per-module mean expression z-scored across spots), aligns to
the embeddings, and writes `<lib>.h5` (embeddings + gene_expression + n_spots) into
work_dir/mil_processed_samples/.

Also writes the reusable `gene_list.pkl` / `modules_info.pkl` so training and
prediction share an identical target definition.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spamil.config import cget, expand
from spamil import io as sio
from spamil.utils import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Target-definition artifacts (gene list / module info)
# --------------------------------------------------------------------------- #
def build_or_load_gene_list(cfg: dict, paths: dict, samples: list) -> list:
    """Return the gene list, building it (top-N HVG over all samples) if needed."""
    existing = cget(cfg, "targets.gene_list")
    out_path = Path(paths["work_dir"]) / "gene_list.pkl"
    if existing:
        with open(expand(existing), "rb") as f:
            return list(pickle.load(f))
    if out_path.exists():
        log.info("Reusing gene list at %s", out_path)
        with open(out_path, "rb") as f:
            return list(pickle.load(f))

    import anndata as ad
    import scanpy as sc

    top_n = int(cget(cfg, "targets.top_n", 1000))
    log.info("Selecting top-%d highly variable genes across %d samples", top_n, len(samples))
    adatas = []
    for lib in samples:
        a = sio.load_expression(lib, cfg)
        a.obs_names = [f"{lib}:{bc}" for bc in a.obs_names]
        adatas.append(a)
    combined = ad.concat(adatas, join="outer", fill_value=0)
    sc.pp.normalize_total(combined, inplace=True)
    sc.pp.log1p(combined)
    sc.pp.highly_variable_genes(combined, n_top_genes=top_n)
    gene_list = combined.var_names[combined.var["highly_variable"]].tolist()

    with open(out_path, "wb") as f:
        pickle.dump(gene_list, f)
    log.info("Saved gene list (%d genes) to %s", len(gene_list), out_path)
    return gene_list


def build_or_load_modules_info(cfg: dict, paths: dict) -> dict:
    """Return module info dict, building it from `targets.modules_csv` if needed."""
    existing = cget(cfg, "targets.modules_info")
    out_path = Path(paths["work_dir"]) / "modules_info.pkl"
    if existing:
        with open(expand(existing), "rb") as f:
            return pickle.load(f)
    if out_path.exists():
        log.info("Reusing module info at %s", out_path)
        with open(out_path, "rb") as f:
            return pickle.load(f)

    modules_csv = cget(cfg, "targets.modules_csv")
    if not modules_csv:
        raise ValueError("targets.modules_csv is required when targets.type == 'modules'")
    df = pd.read_csv(expand(modules_csv))
    module_gene_dict = df.groupby("module")["id"].apply(list).to_dict()
    module_names = sorted(module_gene_dict.keys())
    info = {
        "module_gene_dict": module_gene_dict,
        "module_names": module_names,
        "module_label_list": [f"module_{m}" for m in module_names],
        "n_modules": len(module_names),
    }
    with open(out_path, "wb") as f:
        pickle.dump(info, f)
    log.info("Saved module info (%d modules) to %s", info["n_modules"], out_path)
    return info


# --------------------------------------------------------------------------- #
# Per-sample combined H5 construction
# --------------------------------------------------------------------------- #
def _load_embeddings(paths: dict, lib_id: str) -> np.ndarray:
    emb_file = Path(paths["embeddings"]) / f"{lib_id}_patch_embeddings.h5"
    if not emb_file.exists():
        raise FileNotFoundError(f"Run `spamil preprocess` first; missing {emb_file}")
    with h5py.File(emb_file, "r") as f:
        return f["embeddings"][:]


def _normalised_expr_df(cfg: dict, lib_id: str) -> pd.DataFrame:
    import scanpy as sc
    adata = sio.load_expression(lib_id, cfg)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    return adata.to_df()


def _write_combined(out_file: Path, X: np.ndarray, y: np.ndarray, lib_id: str, extra_attrs: dict):
    n_min = min(X.shape[0], y.shape[0])
    X, y = X[:n_min], y[:n_min]
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("embeddings", data=X)
        hf.create_dataset("gene_expression", data=y)
        hf.attrs["n_spots"] = X.shape[0]
        hf.attrs["library_id"] = lib_id
        for k, v in extra_attrs.items():
            hf.attrs[k] = v
    return X.shape


def process_sample_genes(cfg: dict, paths: dict, lib_id: str, gene_list: list,
                         force: bool) -> bool:
    out_file = Path(paths["processed"]) / f"{lib_id}.h5"
    if out_file.exists() and not force:
        log.info("[%s] target H5 exists, skipping", lib_id)
        return True
    X = _load_embeddings(paths, lib_id)
    expr = _normalised_expr_df(cfg, lib_id)
    available = [g for g in gene_list if g in expr.columns]
    if len(available) < len(gene_list):
        log.warning("[%s] only %d/%d genes present; missing genes set to 0",
                    lib_id, len(available), len(gene_list))
    # Keep full gene_list width/order; fill missing genes with 0.
    y = np.zeros((expr.shape[0], len(gene_list)), dtype=np.float32)
    if available:
        y[:, [gene_list.index(g) for g in available]] = expr[available].values
    shape = _write_combined(out_file, X, y, lib_id, {"n_genes": len(gene_list)})
    log.info("[%s] wrote genes target X=%s y=(%d,%d)", lib_id, shape, shape[0], len(gene_list))
    return True


def process_sample_modules(cfg: dict, paths: dict, lib_id: str, info: dict,
                           force: bool) -> bool:
    from scipy.stats import zscore as scipy_zscore
    out_file = Path(paths["processed"]) / f"{lib_id}.h5"
    if out_file.exists() and not force:
        log.info("[%s] target H5 exists, skipping", lib_id)
        return True
    X = _load_embeddings(paths, lib_id)
    expr = _normalised_expr_df(cfg, lib_id)

    module_scores = pd.DataFrame(index=expr.index)
    for mod in info["module_names"]:
        genes = [g for g in info["module_gene_dict"][mod] if g in expr.columns]
        module_scores[f"module_{mod}"] = expr[genes].mean(axis=1) if genes else 0.0
    module_scores = module_scores.apply(scipy_zscore, axis=0).fillna(0.0)
    y = module_scores.values.astype(np.float32)

    shape = _write_combined(out_file, X, y, lib_id, {"n_modules": info["n_modules"]})
    log.info("[%s] wrote modules target X=%s y=(%d,%d)", lib_id, shape, shape[0], info["n_modules"])
    return True


def run_build_targets(cfg: dict, paths: dict, samples=None, target=None, force=False) -> dict:
    """Driver for `spamil build-targets`. Returns target-definition info."""
    target = target or cget(cfg, "targets.type", "genes")
    if samples is None:
        # Only build targets for samples that already have embeddings.
        emb_dir = Path(paths["embeddings"])
        samples = sorted(p.name.replace("_patch_embeddings.h5", "")
                         for p in emb_dir.glob("*_patch_embeddings.h5")
                         if not p.name.endswith("_image_patch_embeddings.h5"))
    if not samples:
        raise RuntimeError("No samples with embeddings found; run `spamil preprocess` first")

    log.info("Building '%s' targets for %d sample(s)", target, len(samples))
    if target == "genes":
        gene_list = build_or_load_gene_list(cfg, paths, samples)
        for lib in samples:
            process_sample_genes(cfg, paths, lib, gene_list, force)
        return {"target_type": "genes", "target_names": gene_list}
    elif target == "modules":
        info = build_or_load_modules_info(cfg, paths)
        for lib in samples:
            process_sample_modules(cfg, paths, lib, info, force)
        return {"target_type": "modules", "target_names": info["module_label_list"]}
    else:
        raise ValueError(f"Unknown target type '{target}' (expected genes|modules)")

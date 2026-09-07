"""Step 4 (`pathmil predict`): run a trained checkpoint on new samples.

Loads a checkpoint (which carries its architecture + target names) and runs the MIL
model over each sample on two tracks:

  - spot bags  (`<lib>_patch_embeddings.h5`)        -> bag/instance/attention
                                                       predictions for PCC evaluation.
  - image grid (`<lib>_image_patch_embeddings.h5`)  -> a no-gap sliding-window grid
                                                       over the whole tissue, used by
                                                       `pathmil plot` for the fine
                                                       superpixel / single-cell maps.

PCC is computed on the spot track whenever the build-targets combined H5
(`mil_processed_samples/<lib>.h5`) is available; otherwise it is skipped (e.g. for
brand-new data that has no targets yet).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathmil.config import cget
from pathmil.data import MILTestDataset, embedding_sample_info
from pathmil.metrics import calculate_metrics
from pathmil.model import load_checkpoint
from pathmil.train import evaluate
from pathmil.utils import get_logger, get_device

log = get_logger()


def _resolve_checkpoint(cfg, paths, checkpoint, target):
    if checkpoint:
        return Path(checkpoint)
    # Default to the full-training checkpoint for the requested target.
    target = target or cget(cfg, "targets.type", "genes")
    default = Path(paths["models"]) / f"full_{target}" / "model_checkpoint.pth"
    if not default.exists():
        raise FileNotFoundError(
            f"No --checkpoint given and default not found: {default}. "
            f"Train a model first (`pathmil train --mode full`).")
    return default


def _run_track(model, emb_dir, lib, suffix, device, bs, nw):
    """Run the model over one embeddings file; return (bag, inst, attn) or None."""
    from torch.utils.data import DataLoader

    try:
        info = embedding_sample_info(emb_dir, lib, suffix=suffix)
    except FileNotFoundError:
        return None
    loader = DataLoader(MILTestDataset(info), batch_size=bs, shuffle=False, num_workers=nw)
    return evaluate(model, loader, device, with_truth=False)


def _evaluate_pcc(paths, lib, bag_preds, target_names, out_dir):
    """Compute + save PCC against the build-targets ground truth, if available."""
    import h5py

    truth_h5 = Path(paths["processed"]) / f"{lib}.h5"
    if not truth_h5.exists():
        log.info("[%s] no build-targets H5 (%s); skipping PCC", lib, truth_h5.name)
        return
    with h5py.File(truth_h5, "r") as hf:
        truth = hf["gene_expression"][:]
    n = min(truth.shape[0], bag_preds.shape[0])
    metrics = calculate_metrics(truth[:n], bag_preds[:n])
    log.info("[%s] PCC mean %.4f median %.4f MSE %.6f R2 %.4f", lib,
             metrics["mean_pcc"], metrics["median_pcc"], metrics["mse"], metrics["r2"])
    pd.DataFrame({"target": target_names, "pcc": metrics["pcc_per_gene"]}).to_csv(
        out_dir / "target_level_pcc.csv", index=False)
    pd.DataFrame([{
        "sample_id": lib, "mean_pcc": metrics["mean_pcc"],
        "median_pcc": metrics["median_pcc"], "std_pcc": metrics["std_pcc"],
        "mse": metrics["mse"], "r2": metrics["r2"],
        "n_spots": n, "n_outputs": len(target_names),
    }]).to_csv(out_dir / "results_summary.csv", index=False)


def run_predict(cfg: dict, paths: dict, samples=None, checkpoint=None, target=None,
                force=False) -> None:
    device = get_device()
    ckpt_path = _resolve_checkpoint(cfg, paths, checkpoint, target)
    model, ckpt = load_checkpoint(ckpt_path, device=device, build_cfg=cfg)
    target_names = ckpt.get("target_names", [])
    target_type = ckpt.get("target_type", "genes")
    log.info("Loaded checkpoint %s (%s, %d outputs)", ckpt_path, target_type, len(target_names))

    emb_dir = Path(paths["embeddings"])
    if not samples:
        samples = sorted(p.name.replace("_patch_embeddings.h5", "")
                         for p in emb_dir.glob("*_patch_embeddings.h5")
                         if not p.name.endswith("_image_patch_embeddings.h5"))
    if not samples:
        raise RuntimeError("No samples with embeddings found; run `pathmil preprocess` first")

    bs = int(cget(cfg, "train.batch_size", 16))
    nw = int(cget(cfg, "train.num_workers", 4))
    for lib in samples:
        out_dir = Path(paths["predictions"]) / lib
        if (out_dir / "bag_predictions.npy").exists() and not force:
            log.info("[%s] predictions exist, skipping", lib)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": range(len(target_names)), "target": target_names}).to_csv(
            out_dir / "target_names.csv", index=False)

        # Track 1: spot bags -> predictions + PCC evaluation.
        spot = _run_track(model, emb_dir, lib, "_patch_embeddings.h5", device, bs, nw)
        if spot is None:
            log.warning("[%s] no spot embeddings; skipping sample", lib)
            continue
        bag_preds, inst_preds, inst_attn = spot
        np.save(out_dir / "bag_predictions.npy", bag_preds)
        np.save(out_dir / "instance_predictions.npy", inst_preds)
        np.save(out_dir / "instance_attention.npy", inst_attn)
        log.info("[%s] predicted %d spots x %d outputs -> %s",
                 lib, bag_preds.shape[0], bag_preds.shape[1], out_dir)
        _evaluate_pcc(paths, lib, bag_preds, target_names, out_dir)

        # Track 2: no-gap sliding-window image grid -> whole-tissue visualization.
        image = _run_track(model, emb_dir, lib, "_image_patch_embeddings.h5",
                           device, bs, nw)
        if image is None:
            log.info("[%s] no image-patch embeddings; visualization track skipped "
                     "(re-run `pathmil preprocess` to generate them)", lib)
            continue
        img_bag, img_inst, img_attn = image
        np.save(out_dir / "image_bag_predictions.npy", img_bag)
        np.save(out_dir / "image_instance_predictions.npy", img_inst)
        np.save(out_dir / "image_instance_attention.npy", img_attn)
        log.info("[%s] predicted %d image patches x %d outputs (visualization)",
                 lib, img_bag.shape[0], img_bag.shape[1])

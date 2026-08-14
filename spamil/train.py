"""Step 3 (`spamil train`): train the attention-MIL regressor.

Two modes (ported from train_mil_loo.py / train_mil_loo_test.py):
  - full : train on every processed sample, save a single checkpoint for prediction.
  - loo  : leave out one sample (by --sample-index), train on the rest, evaluate on
           the held-out sample (which has ground truth) and report PCC/MSE/R2.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spamil.config import cget
from spamil.data import MILRegressionDiskDataset, load_sample_info_from_directory
from spamil.metrics import calculate_metrics
from spamil.model import build_model, check_activation_matches_target, save_checkpoint
from spamil.utils import get_logger, get_device, set_seed

log = get_logger()


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    import torch
    model.train()
    epoch_loss, n = 0.0, 0
    for xb, yb, _ in tqdm(dataloader, desc="Training", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return epoch_loss / max(n, 1)


def evaluate(model, dataloader, device, with_truth=True):
    import torch
    model.eval()
    bag, inst, attn, truth = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            xb = batch[0].to(device)
            bo, io_, at = model(xb, return_instance_predictions=True)
            bag.append(bo.cpu().numpy())
            inst.append(io_.cpu().numpy())
            attn.append(at.cpu().numpy())
            if with_truth:
                truth.append(batch[1].numpy())
    out = (np.concatenate(bag), np.concatenate(inst), np.concatenate(attn))
    if with_truth:
        return out + (np.concatenate(truth),)
    return out


def load_target_names(cfg: dict, paths: dict, target: str, n_outputs: int) -> tuple:
    """Resolve (target_type, target_names) from build-targets artifacts."""
    work = Path(paths["work_dir"])
    if target == "modules":
        info_p = work / "modules_info.pkl"
        if info_p.exists():
            with open(info_p, "rb") as f:
                return "modules", list(pickle.load(f)["module_label_list"])
        return "modules", [f"module_{i}" for i in range(n_outputs)]
    gl = work / "gene_list.pkl"
    if gl.exists():
        with open(gl, "rb") as f:
            return "genes", list(pickle.load(f))
    return "genes", [f"gene_{i}" for i in range(n_outputs)]


def _make_loaders(cfg, dataset, train_idx, eval_idx=None):
    from torch.utils.data import DataLoader, Subset
    bs = int(cget(cfg, "train.batch_size", 16))
    nw = int(cget(cfg, "train.num_workers", 4))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs,
                              shuffle=True, num_workers=nw)
    eval_loader = None
    if eval_idx is not None:
        eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=bs,
                                 shuffle=False, num_workers=nw)
    return train_loader, eval_loader


def _resolve_bias_init(cfg, train_dataset):
    """Where the instance head starts, in target units.

    `softplus(0) = 0.693` against a log1p-normalised target mean of ~0.016 is a
    ~43x overshoot on every output column, so a non-negative head that starts at
    the torch default spends its first epochs walking the bias down.
    """
    if not cget(cfg, "model.bias_init_from_data", True):
        return cget(cfg, "model.output_bias_init")
    return train_dataset.target_mean()


def _apply_bias_init(cfg, train_dataset):
    """Resolve the bias init and write it back into `cfg`, where build_model reads it."""
    bias_init = _resolve_bias_init(cfg, train_dataset)
    cfg.setdefault("model", {})["output_bias_init"] = bias_init
    if bias_init is not None:
        log.info("Initialising output bias so the head starts at %.6g", bias_init)
    return bias_init


def _peek_scale(h5_path) -> dict:
    """Patch geometry the bags were built at, read from the combined H5.

    Taken from the data rather than the config, so a checkpoint cannot claim a
    scale its embeddings were not cut at. Absent for H5s built before
    `preprocess` recorded it.
    """
    import h5py
    with h5py.File(h5_path, "r") as hf:
        return {k: float(hf.attrs[k]) for k in ("target_mpp", "fov_um") if k in hf.attrs}


def _provenance(cfg, train_lib_ids, scale=None, **extra) -> dict:
    """What a checkpoint must carry to still be readable a year from now.

    Without this a checkpoint cannot say what scale its targets were baked at,
    what physical field of view it was shown, which samples it saw, or with which
    seed. The *factors* are stored rather than a label: a label is a naming
    convention and can drift, the factors cannot. `output_activation` and
    `attention_mode` are already in the checkpoint's `config` block, written by
    `save_checkpoint`.

    `scale` comes from the training H5s, not the config -- see `_peek_scale`.
    """
    return {
        "target_sum": float(cget(cfg, "targets.target_sum", 1e4)),
        "output_bias_init": cget(cfg, "model.output_bias_init"),
        "seed": int(cget(cfg, "train.seed", 0)),
        "epochs": int(cget(cfg, "train.epochs", 5)),
        "lr": float(cget(cfg, "train.lr", 1e-4)),
        "batch_size": int(cget(cfg, "train.batch_size", 16)),
        "n_train_samples": len(train_lib_ids),
        "train_lib_ids": list(train_lib_ids),
        **(scale or {}),
        **extra,
    }


def _build_optim(cfg, model):
    import torch.nn as nn
    import torch.optim as optim
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(cget(cfg, "train.lr", 1e-4)),
                            weight_decay=float(cget(cfg, "train.weight_decay", 1e-4)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=5)
    return criterion, optimizer, scheduler


def run_train(cfg: dict, paths: dict, mode=None, target=None, sample_index=None,
              force=False) -> None:
    mode = mode or cget(cfg, "train.mode", "full")
    target = target or cget(cfg, "targets.type", "genes")
    set_seed(int(cget(cfg, "train.seed", 0)))
    device = get_device()
    epochs = int(cget(cfg, "train.epochs", 5))
    grad_clip = float(cget(cfg, "train.grad_clip", 1.0))

    processed_dir = Path(paths["processed"])
    sample_info = load_sample_info_from_directory(processed_dir)
    if not sample_info:
        raise RuntimeError(f"No processed H5 files in {processed_dir}; run `spamil build-targets`")
    dataset = MILRegressionDiskDataset(sample_info)
    n_outputs = _peek_n_outputs(sample_info[0]["file_path"])
    target_type, target_names = load_target_names(cfg, paths, target, n_outputs)
    check_activation_matches_target(cfg, target_type)
    log.info("Mode=%s target=%s | %d samples, %d spots, %d outputs, device=%s",
             mode, target, len(sample_info), len(dataset), n_outputs, device)

    if mode == "full":
        _run_full(cfg, paths, dataset, sample_info, n_outputs, target_type, target_names,
                  device, epochs, grad_clip)
    elif mode == "loo":
        if sample_index is None:
            raise ValueError("--sample-index is required for --mode loo")
        _run_loo(cfg, paths, dataset, sample_info, sample_index, n_outputs,
                 target_type, target_names, device, epochs, grad_clip)
    else:
        raise ValueError(f"Unknown train mode '{mode}' (expected full|loo)")


def _peek_n_outputs(h5_path) -> int:
    import h5py
    with h5py.File(h5_path, "r") as hf:
        return int(hf["gene_expression"].shape[1])


def _train_loop(cfg, model, train_loader, device, epochs, grad_clip):
    criterion, optimizer, scheduler = _build_optim(cfg, model)
    losses = []
    for ep in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        scheduler.step(loss)
        losses.append(loss)
        log.info("Epoch %d/%d - train loss %.6f", ep + 1, epochs, loss)
    return optimizer, losses


def _save_losses(out_dir: Path, losses):
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "train_loss": losses}).to_csv(
        out_dir / "training_losses.csv", index=False)


def _run_full(cfg, paths, dataset, sample_info, n_outputs, target_type, target_names,
              device, epochs, grad_clip):
    train_libs = [s["library_id"] for s in sample_info]
    scale = _peek_scale(sample_info[0]["file_path"])
    train_loader, _ = _make_loaders(cfg, dataset, np.arange(len(dataset)))
    # Every sample is a training sample here, so the whole dataset is the right
    # source for the bias init.
    _apply_bias_init(cfg, dataset)
    model = build_model(cfg, n_outputs).to(device)
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))
    optimizer, losses = _train_loop(cfg, model, train_loader, device, epochs, grad_clip)

    out_dir = Path(paths["models"]) / f"full_{target_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_losses(out_dir, losses)
    ckpt = out_dir / "model_checkpoint.pth"
    save_checkpoint(ckpt, model, optimizer, target_type=target_type,
                    target_names=target_names, train_losses=losses,
                    extra=_provenance(cfg, train_libs, scale=scale, mode="full"))
    log.info("Saved full-training checkpoint -> %s", ckpt)


def _run_loo(cfg, paths, dataset, sample_info, sample_index, n_outputs,
             target_type, target_names, device, epochs, grad_clip):
    lib_ids = [s["library_id"] for s in sample_info]
    if sample_index < 0 or sample_index >= len(lib_ids):
        raise IndexError(f"--sample-index {sample_index} out of range (0-{len(lib_ids) - 1})")
    test_lib = lib_ids[sample_index]
    log.info("LOO: holding out %s (index %d)", test_lib, sample_index)

    test_idx = dataset.get_sample_indices(test_lib)
    test_set = set(test_idx.tolist())
    train_idx = np.array([i for i in range(len(dataset)) if i not in test_set])
    train_loader, test_loader = _make_loaders(cfg, dataset, train_idx, test_idx)

    # From the TRAINING samples only -- the whole dataset would leak the
    # held-out sample's target mean into where the model starts.
    train_info = [s for s in sample_info if s["library_id"] != test_lib]
    _apply_bias_init(cfg, MILRegressionDiskDataset(train_info))

    model = build_model(cfg, n_outputs).to(device)
    optimizer, losses = _train_loop(cfg, model, train_loader, device, epochs, grad_clip)

    bag_preds, inst_preds, inst_attn, bag_truth = evaluate(model, test_loader, device,
                                                           with_truth=True)
    metrics = calculate_metrics(bag_truth, bag_preds)
    log.info("LOO %s | mean PCC %.4f median %.4f MSE %.6f R2 %.4f", test_lib,
             metrics["mean_pcc"], metrics["median_pcc"], metrics["mse"], metrics["r2"])

    out_dir = Path(paths["models"]) / f"loo_{target_type}" / test_lib
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "bag_predictions.npy", bag_preds)
    np.save(out_dir / "bag_truth.npy", bag_truth)
    np.save(out_dir / "instance_predictions.npy", inst_preds)
    np.save(out_dir / "instance_attention.npy", inst_attn)
    pd.DataFrame({"target": target_names, "pcc": metrics["pcc_per_gene"]}).to_csv(
        out_dir / "target_level_pcc.csv", index=False)
    pd.DataFrame([{
        "sample_id": test_lib, "mean_pcc": metrics["mean_pcc"],
        "median_pcc": metrics["median_pcc"], "std_pcc": metrics["std_pcc"],
        "mse": metrics["mse"], "r2": metrics["r2"],
        "n_test_spots": len(test_idx), "n_train_spots": len(train_idx),
        "n_outputs": len(target_names),
    }]).to_csv(out_dir / "results_summary.csv", index=False)
    _save_losses(out_dir, losses)
    save_checkpoint(out_dir / "model_checkpoint.pth", model, optimizer,
                    target_type=target_type, target_names=target_names,
                    train_losses=losses, metrics=metrics,
                    extra=_provenance(cfg, [s["library_id"] for s in train_info],
                                      scale=_peek_scale(train_info[0]["file_path"]),
                                      mode="loo", test_lib_id=test_lib))
    log.info("Saved LOO results -> %s", out_dir)

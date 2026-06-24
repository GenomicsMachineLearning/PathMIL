"""Attention-MIL regressor (ported verbatim from train_mil_loo.py) plus
checkpoint helpers. The same architecture serves both the gene and module
targets; only the output dimension (`num_outputs`) changes."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class MILAttentionRegressor(nn.Module):
    """Multi-Instance Learning model for per-spot expression regression.

    A spot is a "bag" of `n_instances` patch-token embeddings; an attention
    pooling aggregates them into a bag representation that is regressed to the
    target vector (genes or module scores).
    """

    def __init__(self,
                 input_dim: int = 1280,
                 hidden_dim: int = 512,
                 attention_dim: int = 256,
                 num_outputs: int = 1000,
                 dropout_rate: float = 0.3):
        super().__init__()

        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.instance_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

        self.bag_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

    def forward(self, x: torch.Tensor, return_instance_predictions: bool = False):
        batch_size, n_instances, feature_dim = x.shape
        x_flat = x.view(-1, feature_dim)

        H_flat = self.instance_encoder(x_flat)
        H = H_flat.view(batch_size, n_instances, -1)

        A_flat = self.attention(H_flat)
        A = A_flat.view(batch_size, n_instances, 1)
        A = torch.softmax(A, dim=1)

        M = torch.sum(A * H, dim=1)
        bag_out = self.bag_regressor(M)

        if return_instance_predictions:
            inst_out_flat = self.instance_regressor(H_flat)
            inst_out = inst_out_flat.view(batch_size, n_instances, -1)
            return bag_out, inst_out, A.squeeze(-1)

        return bag_out


def build_model(cfg: dict, num_outputs: int) -> MILAttentionRegressor:
    """Construct a model from the `model` block of a merged config."""
    m = cfg.get("model", {})
    return MILAttentionRegressor(
        input_dim=int(m.get("input_dim", 1280)),
        hidden_dim=int(m.get("hidden_dim", 512)),
        attention_dim=int(m.get("attention_dim", 256)),
        num_outputs=int(num_outputs),
        dropout_rate=float(m.get("dropout", 0.3)),
    )


def save_checkpoint(path, model, optimizer, *, target_type, target_names,
                    train_losses, metrics=None, extra=None):
    """Persist a self-describing checkpoint (architecture + target metadata)."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "train_losses": list(train_losses or []),
        "test_metrics": metrics,
        "target_type": target_type,          # "genes" | "modules"
        "target_names": list(target_names),  # gene names or module labels
        "num_outputs": len(target_names),
        "config": {
            "input_dim": model.instance_encoder[0].in_features,
            "hidden_dim": model.instance_encoder[0].out_features,
            "attention_dim": model.attention[0].out_features,
            "num_outputs": len(target_names),
        },
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device=None, build_cfg: dict | None = None):
    """Load a checkpoint and reconstruct the model on `device`.

    Returns (model, ckpt_dict). Architecture is taken from the checkpoint's
    stored config so the loaded weights always match.
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    c = ckpt["config"]
    model = MILAttentionRegressor(
        input_dim=int(c["input_dim"]),
        hidden_dim=int(c["hidden_dim"]),
        attention_dim=int(c["attention_dim"]),
        num_outputs=int(c["num_outputs"]),
        dropout_rate=float((build_cfg or {}).get("model", {}).get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model = model.to(device)
    return model, ckpt

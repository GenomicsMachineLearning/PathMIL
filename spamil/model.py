"""Attention-based **additive** MIL regressor plus checkpoint helpers.

Each spot is a bag of `n_instances` patch-token embeddings. A single regressor
head scores every instance, and the bag prediction is the attention-weighted sum
of those per-instance predictions (additive MIL, Javed et al. 2022). This ties the
bag and instance heads to the *same* supervised weights, so the per-instance
predictions are a genuine, interpretable decomposition of the bag prediction --
unlike the earlier design, which had a separate bag head and left the instance
head untrained (it never entered the loss graph, so it stayed at random init).

The same architecture serves both the gene and module targets; only the output
dimension (`num_outputs`) changes.

The instance head's output activation is configurable (`output_activation`).
`softplus` is the shipped default; `linear` is the historical behaviour and stays
the behaviour of every checkpoint written before that option existed. The
non-negative activations (`softplus`, `relu`, `exp`) exist because real expression
is never negative while an unconstrained head emits negatives at both the instance
and bag level. They are applied to the *instance* head only: attention is a softmax
over instances, so the bag is a convex combination of instance predictions and
inherits non-negativity for free -- while `bag == sum(A * inst)` keeps holding.
Applying the constraint to the bag output instead would break that identity.
Module targets are z-scored across spots and are legitimately signed, so they
require `linear` -- see `check_activation_matches_target`.

The attention shape is configurable too (`attention_mode`). `per_gene` is the
shipped default: every target gets its own softmax over instances,
`A: (B, N, G)`. `shared` is the historical shape -- one weight per instance,
`A: (B, N, 1)`, applied to every target alike. Both keep the two invariants above
-- the softmax is over instances either way, so each target's bag stays a convex
combination of that target's instance predictions. The trade-off is supervision,
not correctness: `dL/d inst[i,g] = dL/d bag[g] * A[i,g]`, so under `per_gene` an
instance prediction is near-unsupervised wherever its target's attention is ~0,
which is most of an N-way softmax."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

#: Supported instance-head output activations.
OUTPUT_ACTIVATIONS = ("linear", "softplus", "relu", "exp")

#: Supported attention shapes. `shared` gives one weight per instance for all
#: targets -- `A: (B, N, 1)`. `per_gene` gives every target its own softmax over
#: instances -- `A: (B, N, G)`. Both keep `bag == sum(A*inst)` and both keep the
#: bag a convex combination per target, so the non-negativity a constrained
#: instance head provides survives either way.
ATTENTION_MODES = ("shared", "per_gene")


def _activation_module(name: str) -> nn.Module | None:
    if name == "linear":
        return None
    if name == "softplus":
        return nn.Softplus()
    if name == "relu":
        return nn.ReLU()
    if name == "exp":
        return _Exp()
    raise ValueError(
        f"Unknown output_activation '{name}' (expected one of {OUTPUT_ACTIVATIONS})")


class _Exp(nn.Module):
    """exp() as a module, so it can live inside the `instance_regressor` Sequential."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


def _inverse_activation(name: str, value: float) -> float:
    """Pre-activation `b` such that `activation(b) == value`.

    Used to start the head at the data mean. `softplus(0) == 0.693` against a
    typical log1p-normalised target mean of ~0.016 is a ~43x overshoot, repeated
    across every output column.
    """
    if name == "linear":
        return value
    if name == "relu":
        return value
    if name == "softplus":
        return math.log(math.expm1(value)) if value > 0 else -20.0
    if name == "exp":
        return math.log(value) if value > 0 else -20.0
    raise ValueError(
        f"Unknown output_activation '{name}' (expected one of {OUTPUT_ACTIVATIONS})")


class MILAttentionRegressor(nn.Module):
    """Multi-Instance Learning model for per-spot expression regression.

    A spot is a "bag" of `n_instances` patch-token embeddings. Every instance is
    scored by `instance_regressor`; the bag prediction is the attention-weighted
    sum of those instance predictions (additive MIL). Both the bag- and
    instance-level outputs therefore flow through the same supervised head.
    """

    def __init__(self,
                 input_dim: int = 1280,
                 hidden_dim: int = 512,
                 attention_dim: int = 256,
                 num_outputs: int = 1000,
                 dropout_rate: float = 0.3,
                 output_activation: str = "linear",
                 output_bias_init: float | None = None,
                 attention_mode: str = "shared"):
        super().__init__()
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation '{output_activation}' "
                f"(expected one of {OUTPUT_ACTIVATIONS})")
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"Unknown attention_mode '{attention_mode}' "
                f"(expected one of {ATTENTION_MODES})")
        self.output_activation = output_activation
        self.attention_mode = attention_mode

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

        # `per_gene` widens this final layer from 1 to num_outputs -- the whole
        # parameter cost of per-gene attention.
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1 if attention_mode == "shared" else num_outputs),
        )

        # Single shared head: scores each instance; the bag prediction is the
        # attention-weighted sum of these per-instance predictions (additive MIL).
        final = nn.Linear(hidden_dim // 2, num_outputs)
        if output_bias_init is not None:
            nn.init.constant_(final.bias,
                              _inverse_activation(output_activation, output_bias_init))
        layers = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            final,
        ]
        act = _activation_module(output_activation)
        if act is not None:
            layers.append(act)
        self.instance_regressor = nn.Sequential(*layers)

    @property
    def output_layer(self) -> nn.Linear:
        """The final Linear of the instance head.

        Named handle, because the layer's index inside the Sequential moves with
        the output activation and callers need to reach it. Deliberately a
        *property* and not an attribute: assigning the module would register it a
        second time, so `state_dict()` would carry both `instance_regressor.N.*`
        and `output_layer.*` for one set of weights, and a strict
        `load_state_dict` would then reject every checkpoint written before the
        alias existed.
        """
        for layer in reversed(self.instance_regressor):
            if isinstance(layer, nn.Linear):
                return layer
        raise AttributeError("instance_regressor has no Linear layer")

    def forward(self, x: torch.Tensor, return_instance_predictions: bool = False):
        batch_size, n_instances, feature_dim = x.shape
        x_flat = x.view(-1, feature_dim)

        H_flat = self.instance_encoder(x_flat)

        A_flat = self.attention(H_flat)
        # (B, N, 1) under `shared`, (B, N, num_out) under `per_gene`. The softmax
        # is over N either way, so every target's weights sum to 1 independently.
        A = A_flat.view(batch_size, n_instances, -1)
        A = torch.softmax(A, dim=1)

        inst_out_flat = self.instance_regressor(H_flat)
        inst_out = inst_out_flat.view(batch_size, n_instances, -1)   # (B, N, num_out)

        # Additive MIL: bag prediction = attention-weighted sum of instance
        # predictions. The bag-level MSE loss thus supervises the instance head.
        # Broadcast under `shared`, elementwise under `per_gene`.
        bag_out = torch.sum(A * inst_out, dim=1)          # (B, num_out)

        if return_instance_predictions:
            # Squeeze only under `shared`, so a 1-output `per_gene` model still
            # reports a (B, N, G) attention rather than collapsing to (B, N).
            attn = A.squeeze(-1) if self.attention_mode == "shared" else A
            return bag_out, inst_out, attn

        return bag_out

    def select_outputs(self, indices) -> "MILAttentionRegressor":
        """Narrow the model to a subset of output columns, in place.

        Inference only -- this rewrites parameters, so it must not be called on a
        model that is still training. Exists because the instance-level outputs
        over a whole tissue at every target run to tens of GB, while a figure
        needs a handful of targets.

        Exactly equivalent to computing every output and indexing the result:
        under `per_gene` each target's softmax over instances is independent of
        the others, so dropping columns cannot change the ones that remain.
        """
        idx = torch.as_tensor(list(indices), dtype=torch.long,
                              device=self.output_layer.weight.device)

        def _narrow(layer: nn.Linear) -> None:
            layer.weight = nn.Parameter(layer.weight.data[idx].clone())
            layer.bias = nn.Parameter(layer.bias.data[idx].clone())
            layer.out_features = int(idx.numel())

        _narrow(self.output_layer)
        if self.attention_mode == "per_gene":
            _narrow(self.attention[-1])
        return self


def check_activation_matches_target(cfg: dict, target_type: str) -> None:
    """Reject a non-negative head on a target that is legitimately signed.

    `spamil/targets.py` z-scores module scores across spots, so roughly half of
    every modules-mode target is negative by construction. Constraining the head
    to be non-negative there would train it to emit ~0 for half the data. Genes
    mode is the only place the constraint is meaningful.
    """
    activation = str(cfg.get("model", {}).get("output_activation", "linear"))
    if target_type == "modules" and activation != "linear":
        raise ValueError(
            f"model.output_activation='{activation}' is invalid for targets.type='modules': "
            "module scores are z-scored across spots and are legitimately signed. "
            "Non-negative heads apply to genes mode only.")


def build_model(cfg: dict, num_outputs: int) -> MILAttentionRegressor:
    """Construct a model from the `model` block of a merged config."""
    m = cfg.get("model", {})
    bias_init = m.get("output_bias_init")
    return MILAttentionRegressor(
        input_dim=int(m.get("input_dim", 1280)),
        hidden_dim=int(m.get("hidden_dim", 512)),
        attention_dim=int(m.get("attention_dim", 256)),
        num_outputs=int(num_outputs),
        dropout_rate=float(m.get("dropout", 0.3)),
        output_activation=str(m.get("output_activation", "linear")),
        output_bias_init=None if bias_init is None else float(bias_init),
        attention_mode=str(m.get("attention_mode", "shared")),
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
            # softplus/relu/exp are parameterless, so without this key a
            # constrained-head checkpoint would load silently into a linear model
            # and emit wrong values with no error.
            "output_activation": getattr(model, "output_activation", "linear"),
            # Unlike the activation, a missing attention_mode would fail loudly on
            # a shape mismatch -- but it also names the shape every downstream
            # attention array has, so it is recorded rather than inferred.
            "attention_mode": getattr(model, "attention_mode", "shared"),
        },
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _canonical_state_dict(state: dict) -> dict:
    """Drop the redundant `output_layer.*` alias some checkpoints carry.

    An earlier version held the instance head's final Linear as an attribute as
    well as inside `instance_regressor`, so its weights were written twice under
    two names. They are the same tensors; keeping only the canonical
    `instance_regressor.*` entries lets one loader read checkpoints from before
    the alias, from while it existed, and from after it was removed.
    """
    return {k: v for k, v in state.items() if not k.startswith("output_layer.")}


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
        # Checkpoints predating the configurable head carry no activation key;
        # they were all linear, so that is the only safe default.
        output_activation=str(c.get("output_activation", "linear")),
        # Likewise for attention: everything written before per-gene attention
        # existed is `shared`.
        attention_mode=str(c.get("attention_mode", "shared")),
    )
    model.load_state_dict(_canonical_state_dict(ckpt["model_state_dict"]))
    if device is not None:
        model = model.to(device)
    return model, ckpt

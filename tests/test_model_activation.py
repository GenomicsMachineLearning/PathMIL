"""Seam: output activation, the additive-MIL identity, and checkpoint round-trip.

The instance head's output activation is configurable. softplus/relu/exp are
parameterless, so a constrained-head checkpoint's state_dict stays byte-compatible
with a linear-head model -- it would load silently and emit wrong values. These
tests pin both the numerical contract (non-negativity, identity preserved) and the
metadata contract that prevents the silent load.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spamil.model import (
    _activation_module,
    MILAttentionRegressor,
    build_model,
    load_checkpoint,
    save_checkpoint,
)

NONNEG_ACTIVATIONS = ("softplus", "relu", "exp")
ALL_ACTIVATIONS = ("linear",) + NONNEG_ACTIVATIONS


def _bag(n_out=7, activation="linear", seed=0, attention_mode="shared"):
    torch.manual_seed(seed)
    model = MILAttentionRegressor(input_dim=16, hidden_dim=8, attention_dim=4,
                                  num_outputs=n_out, dropout_rate=0.0,
                                  output_activation=activation,
                                  attention_mode=attention_mode)
    return model.eval()


class TestOutputActivation:
    def test_linear_head_can_emit_negative_predictions(self):
        # The historical behaviour, and the reason the constrained heads exist:
        # with a linear head some outputs go negative on arbitrary input.
        model = _bag(activation="linear")
        x = torch.randn(4, 12, 16) * 5.0
        bag, inst, _ = model(x, return_instance_predictions=True)
        assert (inst < 0).any()

    @pytest.mark.parametrize("activation", NONNEG_ACTIVATIONS)
    def test_constrained_heads_never_emit_negative_instances(self, activation):
        model = _bag(activation=activation)
        x = torch.randn(4, 12, 16) * 5.0
        _, inst, _ = model(x, return_instance_predictions=True)
        assert (inst >= 0).all()

    @pytest.mark.parametrize("activation", NONNEG_ACTIVATIONS)
    def test_bag_is_nonnegative_whenever_instances_are(self, activation):
        # Attention is a softmax over instances, so the bag is a convex
        # combination -- constraining the instance head is sufficient, and this
        # is why the constraint is NOT applied to bag_out directly.
        model = _bag(activation=activation)
        x = torch.randn(4, 12, 16) * 5.0
        bag, _, _ = model(x, return_instance_predictions=True)
        assert (bag >= 0).all()

    @pytest.mark.parametrize("activation", ALL_ACTIVATIONS)
    def test_additive_identity_holds_for_every_activation(self, activation):
        # bag == sum(attention * instance). Three verify harnesses check this to
        # ~1e-7 on real checkpoints; it must survive the activation change.
        model = _bag(activation=activation)
        x = torch.randn(3, 10, 16)
        bag, inst, attn = model(x, return_instance_predictions=True)
        recon = (attn.unsqueeze(-1) * inst).sum(dim=1)
        np.testing.assert_allclose(bag.detach().numpy(), recon.detach().numpy(),
                                   atol=1e-6)

    @pytest.mark.parametrize("activation", ALL_ACTIVATIONS)
    def test_attention_is_a_softmax_over_instances(self, activation):
        model = _bag(activation=activation)
        _, _, attn = model(torch.randn(3, 10, 16), return_instance_predictions=True)
        np.testing.assert_allclose(attn.sum(dim=1).detach().numpy(),
                                   np.ones(3), atol=1e-6)

    def test_unknown_activation_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="output_activation"):
            MILAttentionRegressor(num_outputs=3, output_activation="gelu")

    def test_default_activation_is_linear_so_existing_behaviour_is_unchanged(self):
        assert MILAttentionRegressor(num_outputs=3).output_activation == "linear"


class TestAttentionShape:
    """Seam: `attention_mode`, the shape of the attention tensor.

    `per_gene` gives every target its own softmax over instances, so the two
    invariants the model rides on -- the additive identity, and the
    non-negativity a constrained instance head buys *via* the convex combination
    -- have to be re-established elementwise rather than by broadcast.
    """

    def test_default_is_shared_so_existing_behaviour_is_unchanged(self):
        assert MILAttentionRegressor(num_outputs=3).attention_mode == "shared"

    def test_unknown_attention_mode_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="attention_mode"):
            MILAttentionRegressor(num_outputs=3, attention_mode="gated")

    def test_shared_attention_emits_one_weight_per_instance(self):
        _, _, attn = _bag(n_out=7)(torch.randn(3, 10, 16),
                                   return_instance_predictions=True)
        assert attn.shape == (3, 10)

    def test_per_gene_attention_emits_one_weight_per_instance_per_gene(self):
        model = _bag(n_out=7, attention_mode="per_gene")
        _, _, attn = model(torch.randn(3, 10, 16), return_instance_predictions=True)
        assert attn.shape == (3, 10, 7)

    def test_per_gene_attention_head_widens_to_num_outputs(self):
        # Per-gene attention's entire parameter cost comes from this layer.
        assert _bag(n_out=7, attention_mode="per_gene").attention[-1].out_features == 7
        assert _bag(n_out=7).attention[-1].out_features == 1

    def test_per_gene_attention_sums_to_one_over_instances_for_every_gene(self):
        # Not just in aggregate: each gene gets an INDEPENDENT softmax over N.
        # This is what keeps the bag a convex combination per gene, which is what
        # carries non-negativity across from the instance head.
        model = _bag(n_out=7, attention_mode="per_gene")
        _, _, attn = model(torch.randn(3, 10, 16), return_instance_predictions=True)
        np.testing.assert_allclose(attn.sum(dim=1).detach().numpy(),
                                   np.ones((3, 7)), atol=1e-6)

    @pytest.mark.parametrize("activation", ALL_ACTIVATIONS)
    def test_additive_identity_holds_under_per_gene(self, activation):
        # bag == sum(A * inst), now an elementwise product over (B, N, G).
        model = _bag(activation=activation, attention_mode="per_gene")
        bag, inst, attn = model(torch.randn(3, 10, 16),
                                return_instance_predictions=True)
        recon = (attn * inst).sum(dim=1)
        np.testing.assert_allclose(bag.detach().numpy(), recon.detach().numpy(),
                                   atol=1e-6)

    @pytest.mark.parametrize("activation", NONNEG_ACTIVATIONS)
    def test_bag_stays_nonnegative_under_per_gene(self, activation):
        # The non-negativity guarantee must survive the shape change, otherwise
        # softplus stops buying a non-negative bag.
        model = _bag(activation=activation, attention_mode="per_gene")
        bag, _, _ = model(torch.randn(4, 12, 16) * 5.0,
                          return_instance_predictions=True)
        assert (bag >= 0).all()

    def test_build_model_reads_attention_mode_from_config(self):
        cfg = {"model": {"input_dim": 16, "hidden_dim": 8, "attention_dim": 4,
                         "attention_mode": "per_gene"}}
        assert build_model(cfg, 5).attention_mode == "per_gene"

    def test_build_model_defaults_to_shared(self):
        assert build_model({"model": {"input_dim": 16}}, 5).attention_mode == "shared"


class TestOutputSlicing:
    """Seam: slicing the head to a handful of targets for a figure's predict pass.

    Instance-level outputs over a whole tissue at every target run to tens of GB
    while a figure needs a handful. Slicing must be exactly equivalent to
    computing all outputs and indexing -- under `per_gene` that holds only because
    each target's softmax over instances is independent of the others.
    """

    @pytest.mark.parametrize("attention_mode", ["shared", "per_gene"])
    def test_sliced_model_reproduces_the_selected_columns(self, attention_mode):
        model = _bag(n_out=7, attention_mode=attention_mode)
        x = torch.randn(2, 6, 16)
        full = model(x).detach().numpy()

        idx = [1, 4, 5]
        got = model.select_outputs(idx)(x).detach().numpy()

        np.testing.assert_allclose(got, full[:, idx], atol=1e-6)

    @pytest.mark.parametrize("attention_mode", ["shared", "per_gene"])
    def test_sliced_model_reports_the_reduced_width(self, attention_mode):
        model = _bag(n_out=7, attention_mode=attention_mode).select_outputs([0, 3])
        assert model.output_layer.out_features == 2
        _, inst, _ = model(torch.randn(2, 6, 16), return_instance_predictions=True)
        assert inst.shape == (2, 6, 2)

    def test_slicing_per_gene_attention_also_slices_the_attention_head(self):
        model = _bag(n_out=7, attention_mode="per_gene").select_outputs([0, 3])
        assert model.attention[-1].out_features == 2

    def test_slicing_leaves_shared_attention_untouched(self):
        model = _bag(n_out=7).select_outputs([0, 3])
        assert model.attention[-1].out_features == 1


class TestSoftplusBiasInit:
    @pytest.mark.parametrize("activation", ["linear", "softplus", "exp"])
    def test_bias_init_makes_the_head_start_at_the_target_mean(self, activation):
        # softplus(0) = 0.693, but a log1p-normalised target mean is ~0.016 -- a
        # 43x overshoot on every column. The spec is "the head's output at
        # zero pre-activation input equals the target mean", so assert exactly
        # that rather than how the inverse happens to be computed.
        target_mean = 0.016
        model = MILAttentionRegressor(input_dim=16, hidden_dim=8, attention_dim=4,
                                      num_outputs=5, dropout_rate=0.0,
                                      output_activation=activation,
                                      output_bias_init=target_mean)
        bias = model.output_layer.bias.detach()
        act = _activation_module(activation)
        started_at = bias if act is None else act(bias)
        np.testing.assert_allclose(started_at.numpy(), np.full(5, target_mean),
                                   rtol=1e-5)

    def test_bias_is_left_at_torch_default_when_no_init_requested(self):
        # Omitting the option must not change existing training behaviour.
        torch.manual_seed(0)
        with_opt = MILAttentionRegressor(input_dim=16, hidden_dim=8, attention_dim=4,
                                         num_outputs=5, dropout_rate=0.0,
                                         output_bias_init=None)
        torch.manual_seed(0)
        without = MILAttentionRegressor(input_dim=16, hidden_dim=8, attention_dim=4,
                                        num_outputs=5, dropout_rate=0.0)
        np.testing.assert_array_equal(with_opt.output_layer.bias.detach().numpy(),
                                      without.output_layer.bias.detach().numpy())


class TestBuildModel:
    def test_build_model_reads_activation_from_config(self):
        cfg = {"model": {"input_dim": 16, "hidden_dim": 8, "attention_dim": 4,
                         "output_activation": "softplus"}}
        assert build_model(cfg, 5).output_activation == "softplus"

    def test_build_model_defaults_to_linear(self):
        assert build_model({"model": {"input_dim": 16}}, 5).output_activation == "linear"


class TestCheckpointRoundTrip:
    @pytest.mark.parametrize("activation", ALL_ACTIVATIONS)
    def test_checkpoint_restores_its_activation(self, tmp_path, activation):
        model = _bag(n_out=5, activation=activation)
        path = tmp_path / "ckpt.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        restored, ckpt = load_checkpoint(path)
        assert ckpt["config"]["output_activation"] == activation
        assert restored.output_activation == activation

    def test_restored_model_reproduces_the_original_predictions(self, tmp_path):
        model = _bag(n_out=5, activation="softplus")
        x = torch.randn(2, 6, 16)
        before = model(x).detach().numpy()
        path = tmp_path / "ckpt.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        restored, _ = load_checkpoint(path)
        np.testing.assert_allclose(restored.eval()(x).detach().numpy(), before,
                                   atol=1e-6)

    def test_legacy_checkpoint_without_activation_loads_as_linear(self, tmp_path):
        # Every checkpoint written before this change (the published cohort
        # models) lacks the key and must keep behaving exactly as before.
        model = _bag(n_out=5, activation="linear")
        path = tmp_path / "legacy.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        del payload["config"]["output_activation"]
        torch.save(payload, path)

        restored, _ = load_checkpoint(path)
        assert restored.output_activation == "linear"

    @pytest.mark.parametrize("attention_mode", ["shared", "per_gene"])
    def test_checkpoint_restores_its_attention_mode(self, tmp_path, attention_mode):
        # A per-gene checkpoint loaded as `shared` would fail on a shape mismatch
        # rather than silently -- but only because the attention head's width
        # changes. Pin it anyway: the mode also names the shape every downstream
        # attention array has.
        model = _bag(n_out=5, attention_mode=attention_mode)
        path = tmp_path / "ckpt.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        restored, ckpt = load_checkpoint(path)
        assert ckpt["config"]["attention_mode"] == attention_mode
        assert restored.attention_mode == attention_mode

    def test_per_gene_checkpoint_reproduces_its_predictions(self, tmp_path):
        model = _bag(n_out=5, activation="softplus", attention_mode="per_gene")
        x = torch.randn(2, 6, 16)
        before = model(x).detach().numpy()
        path = tmp_path / "ckpt.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        restored, _ = load_checkpoint(path)
        np.testing.assert_allclose(restored.eval()(x).detach().numpy(), before,
                                   atol=1e-6)

    def test_legacy_checkpoint_without_attention_mode_loads_as_shared(self, tmp_path):
        # Every checkpoint written before per-gene attention existed lacks the
        # key and must keep loading.
        model = _bag(n_out=5)
        path = tmp_path / "legacy.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        del payload["config"]["attention_mode"]
        torch.save(payload, path)

        restored, _ = load_checkpoint(path)
        assert restored.attention_mode == "shared"

    def test_state_dict_holds_the_output_layer_once(self):
        # `output_layer` is a property, not an assigned submodule. If it is ever
        # turned back into an attribute the final Linear is registered twice, and
        # every checkpoint written before that alias stops loading.
        keys = MILAttentionRegressor(input_dim=4, hidden_dim=4, attention_dim=2,
                                     num_outputs=2).state_dict().keys()
        assert not [k for k in keys if k.startswith("output_layer.")]

    def test_checkpoint_without_the_output_layer_alias_loads(self, tmp_path):
        # The shape of every checkpoint written before the alias existed.
        model = _bag(n_out=5)
        path = tmp_path / "pre_alias.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["model_state_dict"] = {
            k: v for k, v in payload["model_state_dict"].items()
            if not k.startswith("output_layer.")}
        torch.save(payload, path)

        restored, _ = load_checkpoint(path)   # must not raise on missing keys
        assert restored.output_layer.out_features == 5

    def test_checkpoint_carrying_the_output_layer_alias_loads(self, tmp_path):
        # The shape written while the alias existed: the same weights under two
        # names. The duplicate must be dropped, not rejected as an unexpected key.
        model = _bag(n_out=5)
        path = tmp_path / "with_alias.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload["model_state_dict"]
        final = max(int(k.split(".")[1]) for k in sd if k.startswith("instance_regressor."))
        sd["output_layer.weight"] = sd[f"instance_regressor.{final}.weight"].clone()
        sd["output_layer.bias"] = sd[f"instance_regressor.{final}.bias"].clone()
        torch.save(payload, path)

        restored, _ = load_checkpoint(path)   # must not raise on unexpected keys
        np.testing.assert_allclose(restored.output_layer.bias.detach().numpy(),
                                   model.output_layer.bias.detach().numpy(), atol=1e-6)

    def test_prediction_space_is_recorded_for_scoring(self, tmp_path):
        # The eval harness needs to know which space to invert from; it must come
        # from the checkpoint, not from whatever config happens to be passed.
        model = _bag(n_out=5, activation="exp")
        path = tmp_path / "ckpt.pth"
        save_checkpoint(path, model, None, target_type="genes",
                        target_names=[f"g{i}" for i in range(5)], train_losses=[],
                        extra={"prediction_space": "rate"})
        _, ckpt = load_checkpoint(path)
        assert ckpt["prediction_space"] == "rate"

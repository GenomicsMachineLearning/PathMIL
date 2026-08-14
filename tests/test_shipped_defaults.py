"""Seam: the configuration a clean checkout actually trains under.

`configs/default.yaml` is the released model. Everything else -- the paper's
numbers, the checkpoints in the wild -- assumes a reader who clones this repo and
runs it gets softplus + per-gene attention + a fixed target scale + a
data-derived output bias. Nothing else in the test suite would notice if an edit
quietly reverted one of those four, because each is individually a valid
configuration.

The modules case is the one that breaks loudly rather than quietly, and it is
here for the opposite reason: the gene default is invalid for module targets, so
`example_modules.yaml` has to override it or the documented quickstart dies.
"""

from __future__ import annotations

import pytest

from spamil.config import DEFAULT_CONFIG_PATH, cget, load_config
from spamil.model import build_model, check_activation_matches_target

CONFIG_DIR = DEFAULT_CONFIG_PATH.parent


class TestDefaultConfigShipsTheAdoptedModel:
    def test_output_activation_is_softplus(self):
        assert cget(load_config(None, []), "model.output_activation") == "softplus"

    def test_attention_is_per_gene(self):
        assert cget(load_config(None, []), "model.attention_mode") == "per_gene"

    def test_output_bias_is_derived_from_the_data(self):
        assert cget(load_config(None, []), "model.bias_init_from_data") is True

    def test_target_scale_is_a_fixed_ten_thousand(self):
        assert cget(load_config(None, []), "targets.target_sum") == pytest.approx(1e4)

    def test_patches_are_cut_at_a_fixed_physical_scale(self):
        # Without this every slide contributes a different amount of tissue per
        # instance, and the model silently sees a different object per scanner.
        assert cget(load_config(None, []), "embed.target_mpp") == pytest.approx(0.2535)

    def test_the_field_of_view_is_about_one_visium_spot(self):
        cfg = load_config(None, [])
        fov_um = cget(cfg, "embed.patch_size") * cget(cfg, "embed.target_mpp")
        assert fov_um == pytest.approx(56.8, abs=0.1)

    def test_the_default_config_builds_the_model_it_describes(self):
        # cget-level assertions would still pass if build_model stopped reading a
        # key. Build the thing and look at it.
        model = build_model(load_config(None, []), num_outputs=7)
        assert model.output_activation == "softplus"
        assert model.attention_mode == "per_gene"
        assert model.attention[-1].out_features == 7


class TestModulesExampleOverridesTheActivation:
    def test_modules_example_pins_a_linear_head(self):
        cfg = load_config(str(CONFIG_DIR / "example_modules.yaml"), [])
        assert cget(cfg, "model.output_activation") == "linear"

    def test_modules_example_passes_the_trainer_guard(self):
        cfg = load_config(str(CONFIG_DIR / "example_modules.yaml"), [])
        check_activation_matches_target(cfg, "modules")   # must not raise

    def test_the_bare_default_would_not_pass_it(self):
        # The reason the override above has to exist. If this ever stops raising,
        # either the default went back to linear or the guard was dropped.
        with pytest.raises(ValueError, match="modules"):
            check_activation_matches_target(load_config(None, []), "modules")

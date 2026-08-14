"""Seam: the scale the training target is baked at.

`build-targets` writes `gene_expression` into `<lib>.h5` once, so the library
normalisation is fixed at build time -- no training flag can change it later.
These tests pin that `targets.target_sum` is the T in `log1p(c / L * T)`, with
hand-worked expected values rather than a re-expression of the implementation.

The regression they guard: `sc.pp.normalize_total` called with no `target_sum`
falls back to *each sample's own median library size*, so every sample lands on a
different target scale. That is survivable for a single-cohort model and not for
a multi-cohort one, where one non-negative head must fit every cohort's
sequencing depth at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spamil import targets as tgt


def _adata(counts):
    """Minimal AnnData whose to_df() carries named genes, as io.load_expression returns."""
    import anndata

    counts = np.asarray(counts, dtype=np.float32)
    n_spots, n_genes = counts.shape
    return anndata.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"spot{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]),
    )


@pytest.fixture
def fake_expression(monkeypatch):
    """Route io.load_expression to a per-library counts table supplied by the test."""
    tables = {}

    def _load(lib_id, cfg):
        return _adata(tables[lib_id])

    monkeypatch.setattr(tgt.sio, "load_expression", _load)
    return tables


class TestTargetSum:
    def test_configured_target_sum_is_the_T_in_log1p_c_over_L_times_T(self, fake_expression):
        # Spot 0 totals 10 counts, spot 1 totals 5. At T=10 the first spot is
        # already there and the second is scaled x2.
        fake_expression["lib"] = [[1.0, 2.0, 7.0],
                                  [0.0, 0.0, 5.0]]
        got = tgt._normalised_expr_df({"targets": {"target_sum": 10.0}}, "lib")
        expected = np.log1p(np.array([[1.0, 2.0, 7.0],
                                      [0.0, 0.0, 10.0]]))
        np.testing.assert_allclose(got.values, expected, rtol=1e-5)

    def test_default_target_sum_is_ten_thousand_not_the_sample_median(self, fake_expression):
        # The old behaviour normalised to the median library size (7.5 here).
        # The default must be a fixed 1e4 instead.
        fake_expression["lib"] = [[1.0, 2.0, 7.0],
                                  [0.0, 0.0, 5.0]]
        got = tgt._normalised_expr_df({}, "lib")
        expected = np.log1p(np.array([[1000.0, 2000.0, 7000.0],
                                      [0.0, 0.0, 10000.0]]))
        np.testing.assert_allclose(got.values, expected, rtol=1e-5)

    def test_same_composition_at_different_depth_gives_the_same_target(self, fake_expression):
        """The whole point of a fixed T, stated as a test.

        Two samples with identical relative composition but a 10x depth
        difference must produce an identical target. Under the median fallback
        they do not: each sample lands on its own median, so the deeper sample's
        target is uniformly larger and the head has to fit both.
        """
        fake_expression["shallow"] = [[1.0, 2.0, 7.0],
                                      [1.0, 2.0, 7.0]]
        fake_expression["deep"] = [[10.0, 20.0, 70.0],
                                   [10.0, 20.0, 70.0]]
        cfg = {"targets": {"target_sum": 1.0e4}}
        shallow = tgt._normalised_expr_df(cfg, "shallow")
        deep = tgt._normalised_expr_df(cfg, "deep")
        np.testing.assert_allclose(shallow.values, deep.values, rtol=1e-5)

    def test_shipped_configs_parse_target_sum_as_a_number(self):
        """YAML 1.1 parses `1.0e4` as a string; only `1.0e+4` is a float.

        Every current reader wraps it in float(), so a string works by accident.
        This pins it so the next reader does not have to know.
        """
        from spamil.config import DEFAULT_CONFIG_PATH, cget, load_config

        # Resolved from the package, not the cwd, so the test does not depend on
        # where pytest was invoked from.
        cfg_dir = DEFAULT_CONFIG_PATH.parent
        for config in (None, cfg_dir / "example.yaml", cfg_dir / "example_modules.yaml"):
            cfg = load_config(str(config) if config else None, [])
            value = cget(cfg, "targets.target_sum")
            assert isinstance(value, (int, float)), \
                f"{config or 'default.yaml'} has target_sum={value!r} ({type(value).__name__})"

    def test_gene_names_survive_normalisation(self, fake_expression):
        # process_sample_genes indexes this frame by gene symbol, so the columns
        # must stay named -- a positional frame would silently misalign targets.
        fake_expression["lib"] = [[1.0, 2.0, 7.0]]
        got = tgt._normalised_expr_df({}, "lib")
        assert list(got.columns) == ["g0", "g1", "g2"]

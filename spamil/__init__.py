"""SpaMIL: attention-based Multiple Instance Learning for predicting gene/module
expression at Visium spots from H&E histology patch embeddings.

Submodules that need heavy deps (torch, sopa, …) are imported lazily so that
config / IO utilities can be used in lightweight environments.
"""

__version__ = "0.1.0"

__all__ = ["MILAttentionRegressor", "__version__"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "MILAttentionRegressor":
        from spamil.model import MILAttentionRegressor
        return MILAttentionRegressor
    raise AttributeError(f"module 'spamil' has no attribute {name!r}")

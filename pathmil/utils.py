"""Small shared helpers: logging, device, seeding, HF login."""

from __future__ import annotations

import logging
import random

import numpy as np


def get_logger(name: str = "pathmil") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hf_login_if_enabled(cfg: dict) -> None:
    """Best-effort huggingface login for gated foundation models."""
    from pathmil.config import cget
    if not cget(cfg, "embed.hf_login", True):
        return
    try:
        from huggingface_hub import login
        login()
    except Exception as exc:  # pragma: no cover - network/auth dependent
        get_logger().warning("huggingface login skipped: %s", exc)

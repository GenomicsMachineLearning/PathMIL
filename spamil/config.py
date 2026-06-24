"""Configuration loading: default.yaml + dataset YAML + CLI overrides.

Replaces the hostname-based hardcoded paths in the original scripts. A merged
config is a plain nested dict; access helpers (`cget`/`cset`) use dotted keys.
"""

from __future__ import annotations

import os
import copy
from pathlib import Path

import yaml

# Packaged default config lives at <repo>/SpaMIL/configs/default.yaml
_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _PKG_DIR.parent / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _coerce(value: str):
    """Best-effort string -> python scalar for CLI overrides."""
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_override(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override path '{dotted_key}' traverses a non-dict at '{k}'")
    node[keys[-1]] = value


def cget(cfg: dict, dotted_key: str, default=None):
    """Read a value by dotted key, e.g. cget(cfg, 'train.epochs')."""
    node = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def cset(cfg: dict, dotted_key: str, value) -> None:
    _apply_override(cfg, dotted_key, value)


def expand(value):
    """Expand environment variables / ~ in string path-like config values."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def load_config(config_path: str | None = None, overrides: list[str] | None = None) -> dict:
    """Load default config, merge dataset YAML, then apply `key=value` overrides.

    Args:
        config_path: optional path to a dataset YAML.
        overrides:   list of strings like ["train.epochs=1", "embed.batch_size=8"].
    """
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    if config_path:
        with open(config_path) as f:
            dataset_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, dataset_cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key=value")
        key, raw = item.split("=", 1)
        _apply_override(cfg, key.strip(), _coerce(raw.strip()))

    return cfg


def resolve_paths(cfg: dict, create: bool = True) -> dict:
    """Resolve work_dir / tmp_dir and the standard sub-directories.

    Returns a dict of resolved `pathlib.Path`s and also stores them back under
    cfg['_paths'] for convenience.
    """
    work_dir = expand(cget(cfg, "paths.work_dir"))
    if not work_dir:
        raise ValueError("paths.work_dir is required (set it in the dataset YAML or via --override)")
    work_dir = Path(work_dir)

    tmp_dir = expand(cget(cfg, "paths.tmp_dir") or "")
    # If $SLURM_TMPDIR is unset, fall back to a tmp dir under work_dir.
    if not tmp_dir or tmp_dir.startswith("$"):
        tmp_dir = str(work_dir / "tmp")
    tmp_dir = Path(tmp_dir)

    paths = {
        "work_dir": work_dir,
        "tmp_dir": tmp_dir,
        "embeddings": work_dir / cget(cfg, "paths.embeddings_subdir", "embeddings"),
        "processed": work_dir / cget(cfg, "paths.processed_subdir", "mil_processed_samples"),
        "models": work_dir / cget(cfg, "paths.models_subdir", "models"),
        "predictions": work_dir / cget(cfg, "paths.predictions_subdir", "predictions"),
        "plots": work_dir / cget(cfg, "paths.plots_subdir", "plots"),
    }

    if create:
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

    cfg["_paths"] = {k: str(v) for k, v in paths.items()}
    return paths

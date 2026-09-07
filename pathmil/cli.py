"""`pathmil` command-line interface.

Steps:
  pathmil preprocess     Visium + H&E  -> patch embeddings (+ zarr archive)
  pathmil build-targets  embeddings + expression -> combined training H5
  pathmil train          train MIL regressor (full | loo)
  pathmil predict        apply a checkpoint to samples
  pathmil plot           spatial maps / PCC summaries

Every step takes --config <yaml> and any number of --override key=value.
"""

from __future__ import annotations

import argparse
import sys

from pathmil import __version__
from pathmil.config import load_config, resolve_paths, cget, cset
from pathmil.utils import get_logger

log = get_logger()


def _split_samples(value):
    if not value:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def _common(p):
    p.add_argument("--config", help="Path to a dataset YAML config")
    p.add_argument("--override", action="append", default=[],
                   metavar="KEY=VALUE", help="Override any config key (repeatable)")
    p.add_argument("--samples", help="Comma-separated library ids (default: all discovered)")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist")


def build_parser():
    parser = argparse.ArgumentParser(prog="pathmil", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"pathmil {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="Generate patch embeddings from Visium + H&E")
    _common(p)
    p.add_argument("--index", type=int, help="0-based index into discovered samples (SLURM arrays)")

    p = sub.add_parser("build-targets", help="Build combined embeddings+target H5 files")
    _common(p)
    p.add_argument("--target", choices=["genes", "modules"], help="Target type")

    p = sub.add_parser("train", help="Train the MIL regressor")
    _common(p)
    p.add_argument("--target", choices=["genes", "modules"], help="Target type")
    p.add_argument("--mode", choices=["full", "loo"], help="full training or leave-one-out CV")
    p.add_argument("--sample-index", type=int, help="Held-out sample index for --mode loo")

    p = sub.add_parser("predict", help="Apply a checkpoint to samples")
    _common(p)
    p.add_argument("--target", choices=["genes", "modules"], help="Target type (for default ckpt)")
    p.add_argument("--checkpoint", help="Path to model_checkpoint.pth")

    p = sub.add_parser("plot", help="Spatial maps / PCC summaries")
    _common(p)
    p.add_argument("--targets", help="Comma-separated target names to render (default: top variable)")
    p.add_argument("--prediction-dir", help="Directory with prediction .npy (default: work predictions)")
    p.add_argument("--n-top", type=int, help="Number of top-variable targets to plot")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.override)

    # Apply step-level CLI flags into the config so downstream code sees one source.
    if getattr(args, "target", None):
        cset(cfg, "targets.type", args.target)
    if getattr(args, "mode", None):
        cset(cfg, "train.mode", args.mode)

    paths = resolve_paths(cfg, create=True)
    samples = _split_samples(getattr(args, "samples", None))

    if args.command == "preprocess":
        from pathmil.embed import run_preprocess
        run_preprocess(cfg, paths, samples=samples, index=args.index, force=args.force)

    elif args.command == "build-targets":
        from pathmil.targets import run_build_targets
        run_build_targets(cfg, paths, samples=samples,
                          target=cget(cfg, "targets.type"), force=args.force)

    elif args.command == "train":
        from pathmil.train import run_train
        run_train(cfg, paths, mode=cget(cfg, "train.mode"),
                  target=cget(cfg, "targets.type"),
                  sample_index=args.sample_index, force=args.force)

    elif args.command == "predict":
        from pathmil.predict import run_predict
        run_predict(cfg, paths, samples=samples, checkpoint=args.checkpoint,
                    target=cget(cfg, "targets.type"), force=args.force)

    elif args.command == "plot":
        from pathmil.plot import run_plot
        run_plot(cfg, paths, samples=samples,
                 targets=_split_samples(args.targets),
                 prediction_dir=args.prediction_dir, n_top=args.n_top)

    else:  # pragma: no cover
        log.error("Unknown command %s", args.command)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SpaMIL

Attention-based **Multiple Instance Learning** for predicting per-spot **gene
expression** and **gene-module scores** from Visium **H&E histology**.

Each Visium spot is a *bag*; the 256 Virchow2 patch-token embeddings around the
spot are the *instances*. An **additive** attention-MIL regressor scores every
instance, and the spot prediction is the attention-weighted sum of those
per-instance predictions — so the instance-level outputs are a genuine
decomposition of the spot prediction, not a separate untied head. The target
(gene expression / module score) is read directly from each sample's
`filtered_feature_bc_matrix.h5` — **no sample-level clinical metadata is
required**, so the pipeline runs on any new Visium dataset out of the box.

## Pipeline

```
preprocess  →  build-targets  →  train  →  predict  →  plot
(GPU)          (CPU)             (GPU)     (GPU)       (CPU)
```

| step | command | output |
|------|---------|--------|
| 1 | `spamil preprocess` | `work/<name>/embeddings/<lib>_patch_embeddings.h5` + `<lib>_image_patch_embeddings.h5` (+ `<lib>.zarr.zip`) |
| 2 | `spamil build-targets` | `work/<name>/mil_processed_samples/<lib>.h5` (embeddings + target) |
| 3 | `spamil train` | `work/<name>/models/{full,loo}_<target>/.../model_checkpoint.pth` |
| 4 | `spamil predict` | `work/<name>/predictions/<lib>/{bag_predictions.npy + target_level_pcc.csv, image_bag_predictions.npy}` |
| 5 | `spamil plot` | `work/<name>/plots/<lib>/<lib>_<target>_4panel.png` |

`predict` runs two tracks: **spot bags** (`bag_predictions.npy`, evaluated with PCC
against the build-targets ground truth when present → `target_level_pcc.csv` /
`results_summary.csv`) and a **no-gap sliding-window image grid**
(`image_bag_predictions.npy` + `image_instance_predictions.npy`) that `plot` uses to
render the fine superpixel / single-cell maps over the whole tissue.

## Predict from an H&E slide

Once a model is trained, `scripts/predict_he.py` applies it to a plain H&E image — no
Visium spots, no expression matrix, no config file. It reads the slide, embeds each
tissue tile with Virchow2, runs the MIL regressor, and writes per-tile scores plus
spatial heatmaps:

```bash
uv pip install -e .[he]

python scripts/predict_he.py \
    --slide tumour.svs \
    --checkpoint model_checkpoint.pth \
    --out results/ \
    --target-mpp 0.2535
```

```
results/tumour/
  predictions.csv          tile_x, tile_y, <one column per target>
  bag_predictions.npy      (n_tiles, n_outputs)
  tile_xy.npy              (n_tiles, 2) — top-left pixel of each tile
  tumour_overview.png      every target, small multiples over the H&E
  tumour_<target>.png      the 6 most spatially-variable targets
```

`--slide` accepts a file or a directory (and is repeatable). Output columns are named
automatically: the checkpoint carries its own target list, so you never have to tell the
script whether it predicts genes or module scores.

**Match the field of view.** The model always sees a 224px tile, so a slide scanned at a
different magnification than the training data shows it the wrong *physical* area.
`--target-mpp` is the microns-per-pixel the model was **trained** at (e.g. `0.2535` for
40x): each tile is read at the size covering that same physical area and resampled to
224px. Omit it only if your slide is already at the training resolution. If the model was
trained at 40x and you run a 20x slide without `--target-mpp`, the predictions will be
against a field of view twice the intended size.

Embeddings are streamed straight into the regressor and never written to disk, so a
whole-slide image needs only a batch's worth of memory rather than the tens of GB its raw
Virchow2 tokens would occupy. Pass `--save-embeddings` if you want them anyway.

### Single-cell prediction (`--sc_pred`)

The MIL head is **additive** — a tile's score is the attention-weighted sum of its 256
per-instance scores — so those 256 Virchow2 tokens are a genuine **16×16 prediction map
inside each tile**, not a by-product. `--sc_pred` exposes that, and pushes it down to
individual nuclei:

```
tile (bag)  →  instance (16×16 tokens)  →  single cell (StarDist nuclei)
```

```bash
python scripts/predict_he.py \
    --slide tumour.svs --checkpoint model_checkpoint.pth --out results/ \
    --target-mpp 0.2535 --sc_pred
```

```
results/tumour/
  cells.csv                     cell_id, x, y, area, attention, <one column per target>
  cells.parquet                 the same + the StarDist nucleus polygon (GeoParquet)
  instance_predictions.npy      (n_tiles, 256, n_targets)
  instance_attention.npy        (n_tiles, 256)
  tumour_<target>_4panel.png    H&E+nuclei | tile | instance | single-cell
  tumour_segmentation.png       full-resolution crop with nucleus outlines (QC)
```

Nuclei are segmented per tile with StarDist `2D_versatile_he`, on a slightly padded crop
so cells near a tile edge are seen whole. A nucleus is then kept by **the tile whose core
contains its centroid** — tile ownership is a partition, so a cell straddling a seam is
detected twice but counted once. Each cell's score is the mean of the instance tokens its
outline overlaps.

Two things to know:

- **Segmentation runs on CPU** and is the slow part (the GPU stays reserved for Virchow2 —
  TensorFlow would otherwise claim the whole device). Expect it to dominate the runtime on
  a whole slide.
- **The instance array is held in memory** as `(n_tiles, 256, n_targets)` so the most
  variable targets can be picked after the pass. That is fine for a module model (tens of
  targets) but not for a 1000-gene one; the script estimates the size up front and tells
  you to pass `--targets` rather than dying halfway. `--max-instance-gb` raises the limit.

StarDist weights are not shipped with this repo: they are downloaded on first use, or
point `--stardist-model-dir` at a local `2D_versatile_he` folder for offline runs.

`openslide-python` needs the OpenSlide **C library** installed
(`conda install -c conda-forge openslide-python`, or `apt install libopenslide0`);
without it the script falls back to PIL/tifffile, which cannot read pyramidal WSI formats
such as `.svs`.

## Install

```bash
mamba create -n SpaMIL python=3.11 uv
mamba activate SpaMIL
git clone https://github.com/GenomicsMachineLearning/SpaMIL.git
cd SpaMIL
uv pip install -e .[embed]
```

The heavy embedding extras (`sopa`, `spatialdata*`, `timm`, `stardist`, …) are only
needed for the `preprocess` step. If you already have embeddings and only want to
`train` / `predict` / `plot`, a plain `pip install -e .` (torch + numpy + scanpy +
h5py) is enough.

> **Note:** SpaMIL is currently designed to be run from a checked-out source tree
> (editable install, `pip install -e .`), so that `configs/default.yaml` resolves
> relative to the package.

The Virchow2 foundation model (`paige-ai/Virchow2`) is gated on Hugging Face — request
access and log in (`huggingface-cli login`) before running `preprocess`. StarDist
downloads its `2D_versatile_he` nucleus model automatically on first use.

## Expected data layouts

Declare the layout in the config (`dataset.layout`) — it is **not** auto-detected.

```
layout: flat                              layout: outs
<data_dir>/                               <data_dir>/
  VLP97_A/                                  VLP97_A/
    filtered_feature_bc_matrix.h5             outs/
    spatial/                                    filtered_feature_bc_matrix.h5
      tissue_positions.csv                      spatial/
      scalefactors_json.json                      tissue_positions.csv
      tissue_hires_image.png                      ...
  VLP97_D/ ...                              VLP97_D/ ...
<he_dir>/                                 (H&E resolved via he_dir/he_pattern)
  VLP97_A1.tif   # he_pattern: "{library_id}1.tif"
```

## Configuration

`configs/default.yaml` holds every parameter with defaults. A dataset YAML is
merged on top, and any key can be overridden on the CLI. Start from the bundled
`configs/example.yaml` (gene targets) or `configs/example_modules.yaml`
(gene-module targets) and edit the paths for your data:

```bash
spamil preprocess --config configs/example.yaml --override embed.batch_size=8
```

## Quickstart

End to end on your own Visium cohort (point `configs/example.yaml` at your data first):

```bash
spamil preprocess    --config configs/example.yaml                       # GPU; patch embeddings
spamil build-targets --config configs/example.yaml --target genes        # CPU; combined H5
spamil train         --config configs/example.yaml --target genes --mode full   # GPU; full training
spamil predict       --config configs/example.yaml --target genes        # GPU; predictions + PCC
spamil plot          --config configs/example.yaml                       # CPU; figures
```

### Gene vs module targets

Everything is driven by `--target {genes,modules}` (no separate scripts). For
modules, point `targets.modules_csv` at a CSV with columns `id` (gene symbol) and
`module` (see `assets/modules/example_gene_modules.csv`), then:

```bash
spamil build-targets --config configs/example_modules.yaml --target modules
spamil train         --config configs/example_modules.yaml --target modules --mode full
```

### Leave-one-out cross-validation

```bash
spamil train --config configs/example.yaml --target genes --mode loo --sample-index 0
```

Each run holds out one sample, trains on the rest, and writes PCC/MSE/R² plus
predictions under `models/loo_<target>/<held-out-lib>/`. Sweep `--sample-index`
over `0 .. N-1` to cover every sample.

## Applying a model trained on other data

Prediction only needs the new samples' embeddings + a checkpoint:

```bash
spamil preprocess --config <new_cfg>
spamil predict   --config <new_cfg> --checkpoint /path/to/model_checkpoint.pth
spamil plot      --config <new_cfg>
```

The checkpoint carries its architecture and target-name list, so predictions are
labelled with the original training targets.

## Onboarding a brand-new dataset

1. Copy `configs/example.yaml`, set `data_dir`, `layout`, `he_dir`, `he_pattern`,
   and `paths.work_dir`.
2. Run the pipeline above.

## License

Released under the [MIT License](LICENSE).

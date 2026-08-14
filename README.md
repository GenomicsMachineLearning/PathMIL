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

Two properties of the shipped model are worth stating up front, since both are
configurable and the defaults are deliberate:

- **Attention is per-target.** Every target gets its own softmax over the 256
  instances (`A: (B, 256, G)`), so a gene can attend where it likes instead of
  sharing one saliency map with every other gene. `model.attention_mode: shared`
  restores the single-map form (`A: (B, 256, 1)`), which is also what every
  checkpoint written before this release resolves to. Per-target attention costs
  memory — see [Per-target attention: the memory cost](#per-target-attention-the-memory-cost).
- **The instance head is non-negative.** It ends in a `softplus`, because
  expression never is negative while an unconstrained head emits negatives at
  both the instance and the spot level. Attention being a softmax, the spot
  prediction is a convex combination of non-negative instance scores and inherits
  non-negativity for free — while `bag == Σ(A · inst)` still holds exactly.

Both properties hold under either attention shape, and both are recorded in the
checkpoint.

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
224px. **Checkpoints trained by this version record it, and the script uses that by
default** — you only need the flag for a checkpoint that predates the record, or to
override it (`--target-mpp 0` tiles at native resolution). If the model was trained at
40x and you run a 20x slide without a target MPP, the predictions will be against a field
of view twice the intended size.

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
  cells.csv                     cell_id, x, y, area, <one column per target>,
                                then attention (shared) or attention_<target> (per-target)
  cells.parquet                 the same + the StarDist nucleus polygon (GeoParquet)
  instance_predictions.npy      (n_tiles, 256, n_targets)
  instance_attention.npy        (n_tiles, 256) shared | (n_tiles, 256, n_targets) per-target
                                — omitted under --no-save-attention, as are the
                                attention columns of the cell table
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
- **Per-target attention doubles that store, and changes the cell table.** With a
  `per_gene` checkpoint each nucleus has one attention weight *per target*, not one
  overall, so `cells.csv` carries an `attention_<target>` column per scored target
  rather than a single `attention` column, and `instance_attention.npy` gains a target
  axis. `--targets` narrows the attention array by the same columns as the scores, so
  it stays the same size as the instance store — `--max-instance-gb` accounts for
  both, and naming fewer targets shrinks both.
- **`--no-save-attention` drops the attention half entirely** when you only want the
  predictions. The attention is then never accumulated, so it costs no memory rather
  than merely no disk: under a `per_gene` checkpoint that roughly halves both the
  `--max-instance-gb` requirement and the output size. Instance scores, cell scores
  and every figure are unchanged; under `shared` attention it only saves a small file.
  Attention is always recomputable from the checkpoint, so dropping it is safe.

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

### The model settings, and why they are not free parameters

Four keys in `configs/default.yaml` define the released model. Each is a valid
knob, but each also has one setting the rest of the pipeline assumes.

| key | default | why |
|---|---|---|
| `targets.target_sum` | `1.0e+4` | The `T` in `log1p(c / L · T)`. **Baked into the H5 at build-targets time** — changing it means re-running `build-targets --force`. With no value, scanpy normalises each sample to *its own median library size*, so every sample lands on a different target scale and one head has to fit every sample's sequencing depth. Note the `+`: YAML 1.1 parses `1.0e4` as a *string*. |
| `model.output_activation` | `softplus` | Constrains the instance head to be non-negative. **Must be `linear` for `targets.type: modules`** — module scores are z-scored across spots and are legitimately signed, so a non-negative head would learn to emit ~0 for half the data. `spamil train` refuses the combination outright; `configs/example_modules.yaml` already sets it. |
| `model.attention_mode` | `per_gene` | One softmax over instances per target. See the memory cost below. |
| `model.bias_init_from_data` | `true` | `softplus(0) = 0.693` against a typical target mean of ~0.016 is a ~43× overshoot on every output column, so the head starts at the training-set target mean instead (stored pre-activation). Under `--mode loo` the mean comes from the **training** samples only, never the held-out one. |
| `embed.target_mpp` | `0.2535` | The physical scale of a patch — see below. |

Checkpoints record all of this. Alongside the architecture and target names, each
`model_checkpoint.pth` carries a provenance block — `target_sum`,
`output_bias_init`, `seed`, `epochs`, `lr`, `batch_size`, `mode` and
`train_lib_ids` — so a checkpoint can say what it was trained on and at what
target scale, rather than relying on whatever config happens to be nearby.

### Patch scale: tiles cover microns, not pixels

A fixed 224 px patch is not a fixed amount of tissue. On a 0.25 µm/px slide it is
56.7 µm across; on a 1.33 µm/px slide it is 298.6 µm — a 5.3× difference in what
the foundation model is actually shown, presented to it as if it were the same.

With `embed.target_mpp` set, each sample instead reads
`round(patch_size × target_mpp / slide_mpp)` **native** pixels around every spot,
and the model transform resizes that to `patch_size`. Every slide then contributes
the same physical field of view per instance — at the default
`224 × 0.2535 = 56.8 µm`, about one Visium spot. Set `embed.target_mpp: null` for
the old behaviour (read `patch_size` native px, no rescaling).

**Where `slide_mpp` comes from matters.** SpaMIL measures it from the Visium spot
pitch: spots adjacent in an array row are 100 µm apart by hardware, expressed in
the same coordinate space the patches are cut in, and robust to registration
(`spamil/scale.py`). It deliberately does **not** use either of the two obvious
alternatives:

- `55 / spot_diameter_fullres` — `spot_diameter_fullres` does not measure the
  55 µm capture spot, and what it measures is not even a fixed convention: across
  four cohorts it works out at 70.0, 71.0, 65.0 and 59.9 µm against the pitch,
  varying by Space Ranger version. Treating it as 55 µm under-estimates MPP by
  9–29%.
- TIFF resolution tags — frequently placeholders (300 dpi, 96 dpi, 1 px = 1 mm).

`preprocess` records `target_mpp`, `slide_mpp`, `fov_um` and `read_px` in each
embeddings H5; `build-targets` carries them into the combined H5; `train` copies
them into the checkpoint **from the data, not the config**, so a checkpoint cannot
claim a scale its embeddings were not cut at. `scripts/predict_he.py` then reads
`--target-mpp` from the checkpoint by default, instead of the user having to
remember it (`--target-mpp 0` forces native resolution).

### Per-target attention: the memory cost

`attention_mode: per_gene` widens the attention head's final layer from `→ 1` to
`→ G`, and widens every attention array it produces by the same factor:

| | `shared` | `per_gene` |
|---|---|---|
| attention head params | `attention_dim × 1` | `attention_dim × G` |
| `instance_attention.npy` | `(n_spots, 256)` | `(n_spots, 256, G)` |
| checkpoint size, G = 4,876 | ~28 MB | ~43 MB |

The attention array therefore becomes **the same size as the instance
predictions**, so `spamil predict` and `spamil train --mode loo` roughly double
both peak RAM and output bytes. `train.evaluate` concatenates per-batch arrays in
memory before writing, so the peak is the whole array: at 2,000 spots × 4,876
targets that is ~10 GB for the attention alone, and the whole-tissue image track
in `spamil predict` is larger still.

Nothing guards this on the `spamil predict` track — the array is written whatever
its size. If it is too large:

- set `model.attention_mode: shared` for a `(n_spots, 256)` attention map;
- reduce `targets.top_n`;
- on the `scripts/predict_he.py` track, pass `--no-save-attention`, which drops the
  array from memory as well as from disk (`--max-instance-gb` guards that track); or
- narrow a loaded model with `MILAttentionRegressor.select_outputs([...])`, which
  is exactly equivalent to computing every output and indexing the result (each
  target's softmax over instances is independent of the others).

The attention array is always recomputable from the checkpoint, so deleting it is
safe.

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

Module scores are z-scored across spots, so roughly half of every target is
negative by construction and the non-negative head the gene default ships would
be wrong. `configs/example_modules.yaml` therefore pins
`model.output_activation: linear`; keep that in any modules config of your own, or
`spamil train` will refuse to start.

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

The checkpoint carries its architecture, target-name list and training
provenance, so predictions are labelled with the original training targets and
the checkpoint can state the target scale it was trained at.

## Upgrading from an earlier version

Two things changed in how a model is trained. Neither affects loading an existing
checkpoint, and one affects existing *data*.

- **Existing checkpoints keep working, unchanged.** `load_checkpoint` defaults a
  missing `output_activation` to `linear` and a missing `attention_mode` to
  `shared`, which is what every earlier checkpoint was. No migration, no
  re-training, no flags.
- **Embeddings should be regenerated if you want the fixed patch scale.**
  `preprocess` now reads patches at a fixed physical size (`embed.target_mpp`,
  default 0.2535 µm/px) rather than a fixed pixel size. Existing embeddings were
  cut at native resolution; they still work, but they are not on the same scale as
  a model trained under the new default, and they carry no `target_mpp` for
  `predict_he.py` to pick up. Set `embed.target_mpp: null` to keep the old
  behaviour instead.
- **Existing processed H5s should be rebuilt.** `build-targets` now normalises
  each sample to a fixed `targets.target_sum` instead of that sample's own median
  library size, and the target is baked into `mil_processed_samples/<lib>.h5` at
  build time. Training on H5s built by an earlier version still runs, but the
  checkpoint's provenance block will record a `target_sum` the data was not built
  with, and mixing old and new H5s in one `work_dir` trains a single head across
  two target scales without any error. Re-run `spamil build-targets --force`
  before training.

## Tests

```bash
pip install -e .[dev]
pytest
```

The suite is pure Python — no GPU, no data, no network. It pins the model
contract (the additive identity `bag == Σ(A·inst)`, non-negativity, attention
summing to 1 over instances under both shapes), the checkpoint compatibility
guarantees above, the bias-init derivation, the target scale, and that
`configs/default.yaml` still ships the configuration described here.

## Onboarding a brand-new dataset

1. Copy `configs/example.yaml`, set `data_dir`, `layout`, `he_dir`, `he_pattern`,
   and `paths.work_dir`.
2. Run the pipeline above.

## License

Released under the [MIT License](LICENSE).

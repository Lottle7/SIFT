# SIFT

SIFT learns transferable urban-region representations from source-city POI and
mobility data. A two-stage procedure trains a source branch and then a POI-only
deployment branch. Target-city representations require POI data only.

This release contains single-source and dual-source transfer among **Beijing
(`bj`), Chengdu (`cd`), and Xi'an (`xa`)**, for population, GDP, and CO2 prediction.

## Installation

Use Python 3.9–3.12 in a separate environment:

```bash
python -m pip install -r requirements.txt
```

The pinned environment was checked locally with Python 3.9 and PyTorch 2.8 on
CPU. Use `--device cpu` for CPU execution or `--device cuda` with a CUDA-enabled
PyTorch installation. The training commands fall back to CPU if CUDA is unavailable.

## Files

```text
model.py              Source branch, POI-only deployment branch, and losses
args.py               Command-line arguments and YAML configuration loading
utils.py              Data loading, preprocessing, and metrics
poi_alignment.py      POI frequency ranking and zero padding
train_stage1.py        Source branch training
train_stage2.py        POI-only deployment branch training
infer.py               Frozen target representation generation
evaluate.py            Downstream task evaluation
configs/default.yaml  Shared defaults for all cities and downstream tasks
dataset/               Preprocessed POI and source mobility inputs
tasks/                 Downstream task labels
```

## Data

The repository includes the preprocessed POI, source mobility, and task-label
arrays for all three cities. **Pretrained weights are not bundled.** The data
are organized as follows, with the original region ordering preserved.

```text
dataset/
  bj/                  # Repeat for cd/ and xa/
    poi_f.npy
    source_matrix_norm.npy
    source_sum.npy
    destina_matrix_norm.npy
    destina_sum.npy
tasks/
  bj/                  # Repeat for cd/ and xa/
    population_sum.npy
    gdp_sum.npy
    co2_sum.npy
```

| File | Shape | Meaning |
| --- | --- | --- |
| `poi_f.npy` | `[N, C]` | Nonnegative POI counts |
| `source_matrix_norm.npy` | `[N, N]` | Row-normalized outgoing OD matrix |
| `source_sum.npy` | `[N, 1]` | Outgoing flow totals; OD is reconstructed by elementwise multiplication |
| `destina_matrix_norm.npy` | `[N, N]` | Row-normalized incoming OD matrix |
| `destina_sum.npy` | `[N, 1]` | Incoming flow totals; multiplication reconstructs the transpose of the outgoing OD matrix |
| `*_sum.npy` in `tasks/` | `[N]` or `[N, 1]` | Region-level task labels |

The original incoming-flow files are included for completeness. The current
loader reconstructs OD from the `source_*` files and computes incoming
adjacency from its transpose, so it does not need to read the `destina_*` files.

The supplied configurations use 594/836/660 regions and 159/162/154 original
POI columns for Beijing/Chengdu/Xi'an. POI columns are sorted by city-wide
frequency and zero-padded to 162 dimensions. Normalization statistics are
computed from source cities.

Source mobility is used by the source branch and to construct transfer supervision.
The deployment branch forward pass uses POI inputs. Training and checkpoint
selection use source POI and mobility without loading downstream task labels.
Target mobility is not loaded. Task labels are read separately by `evaluate.py`.

For data stored elsewhere, append `--data_root /path/to/data_root` to each
command. Training and inference require `dataset/`; downstream evaluation
additionally requires `tasks/`.

## Single-source transfer

Run the following commands from this repository's root. This example transfers
from Chengdu to Beijing. Train once and evaluate the same representation on
population, GDP, and CO2:

```bash
python train_stage1.py --config configs/default.yaml --source_cities cd --target_city bj --device cpu
python train_stage2.py --config configs/default.yaml --source_cities cd --target_city bj --device cpu
```

Generated files are saved outside the source directory, under `../sift_runs/`.
Override the destination with `--output_dir /path/to/runs` on both training
commands. Stage II automatically finds the corresponding Stage-I checkpoint;
`--stage1_checkpoint /path/to/stage1.pt` selects an existing one explicitly.

```bash
python infer.py --checkpoint ../sift_runs/cd_to_bj/stage2/default/stage2_full_local_role_cd_to_bj.pt --city bj --output ../sift_runs/cd_to_bj/embedding.npy
python evaluate.py --embedding ../sift_runs/cd_to_bj/embedding.npy --city bj --task all
```

`infer.py` uses the output representation saved in the checkpoint. For the
default configuration this is `local_role`, the concatenation of local and
role features. Stage II also saves this embedding alongside its checkpoint.

## Dual-source transfer

Use the same configuration with two source cities. For Chengdu + Xi'an → Beijing:

```bash
python train_stage1.py --config configs/default.yaml --source_cities cd xa --target_city bj --device cpu
python train_stage2.py --config configs/default.yaml --source_cities cd xa --target_city bj --device cpu
python infer.py --checkpoint ../sift_runs/cd_xa_to_bj/stage2/default/stage2_full_local_role_cd_xa_to_bj.pt --city bj --output ../sift_runs/cd_xa_to_bj/embedding.npy
python evaluate.py --embedding ../sift_runs/cd_xa_to_bj/embedding.npy --city bj --task all
```

The available directions are:

| Setting | Directions |
| --- | --- |
| Single source | `bj_to_cd`, `bj_to_xa`, `cd_to_bj`, `cd_to_xa`, `xa_to_bj`, `xa_to_cd` |
| Dual source | `cd_xa_to_bj`, `bj_xa_to_cd`, `bj_cd_to_xa` |

All nine directions share `configs/default.yaml`. Only the source/target city
arguments change. Run each training stage once per direction and reuse the
resulting target embedding for all three tasks. The default `full` training
mode includes complete-representation matching as well as structural losses;
`local_role` selects the final local and role blocks. Checkpoint and embedding
paths are printed at the end of training.

## Configuration and evaluation

A YAML file contains `source_cities`, `target_city`, and separate `stage1` and
`stage2` settings. Explicit command-line options override YAML settings, for
example `--seed 42` or `--epochs 400`. Stage-I and Stage-II seeds are recorded
separately. Source cities must be distinct and exclude the target.

The shared defaults follow the paper's stated dimensions and training schedule:

| Setting | Default |
| --- | --- |
| Roles / POI hidden dimension / MLP hidden width | 8 / 96 / 96 |
| Target embedding | 48 local + 24 role = 72 dimensions |
| Stage-I phase lengths | 12 / 24 / 36 epochs |
| Stage-I learning rates | 0.0005 / 0.003 / 0.0002 |
| Alternating operator learning rate | 0.0003 |
| Stage-II duration | 300 epochs, within the paper's 220–400 range |
| Stage-II cosine learning-rate bounds | 0.00085 to 0.00013 |
| Adam weight decay | 0.0001 |

Stage II inherits its architecture from the Stage-I checkpoint. The internal
96-dimensional training representation includes a 24-dimensional interaction
block; the final target representation retains the 48-dimensional local and
24-dimensional role blocks. Changing the hidden width does not change these
output dimensions.

Stage-II checkpoint selection uses the minimum source-city validation objective,
evaluated after every epoch. By default, 20% of each source city's regions are
held out for the region-level validation losses. The objective retains the
training loss weights and city-level regularizers and is averaged equally
across source cities. Representation diagnostics are logged but do not add
extra checkpoint-selection penalties. No downstream Ridge evaluation is run
during training or checkpoint selection.
The shared defaults are provided for running the released model and have not
been validated as a configuration that reproduces the paper's reported scores.

Evaluation follows ordered five-fold cross-validation with `Ridge(alpha=1.0)`.
Negative predictions are clipped to zero. MAE, RMSE, and R² are computed from
the pooled out-of-fold predictions. `--task all` evaluates all three labels on
one embedding. Use `--output /path/to/metrics.json` to save metrics.

Append `--smoke` to both training commands for a short pipeline check. These
runs use a separate `smoke/` location and do not represent full-training results.
All four entry points provide `--help`.


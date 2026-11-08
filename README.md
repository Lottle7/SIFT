# SIFT

PyTorch implementation of SIFT for cross-city urban region representation learning.
SIFT learns from source-city POI and mobility data and generates target-city
representations using POI data only.

## Environment

Use Python 3.9–3.12 and install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

Preprocessed data for Beijing (`bj`), Chengdu (`cd`), and Xi'an (`xa`) are included
in `dataset/`. Population, GDP, and carbon emission labels are provided in `tasks/`.

## Training

Run the following commands from the repository root. The example configuration
in `configs/default.yaml` uses Beijing and Chengdu as sources and Xi'an as the target.

```bash
python train_stage1.py --config configs/default.yaml
python train_stage2.py --config configs/default.yaml
```

Set `source_cities` and `target_city` in the configuration to change the transfer
setting. Use one source city for single-source transfer or two for dual-source
transfer. Append `--device cpu` to run on CPU. Outputs are saved in `../sift_runs/`.

## Evaluation

Evaluate the learned target embeddings on all three downstream tasks:

```bash
python evaluate.py --embedding ../sift_runs/bj_cd_to_xa/stage2/default/embedding_full_local_role_xa.npy --city xa --task all
```

The same embeddings are used for population, GDP, and carbon emission prediction
with five-fold Ridge regression.


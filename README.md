# SpaceCast

[![arXiv](https://img.shields.io/badge/arXiv-2509.19605-b31b1b.svg)](https://arxiv.org/abs/2509.19605) [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/deinal/spacecast-data) [![Linting](https://github.com/fmihpc/spacecast/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/fmihpc/spacecast/actions/workflows/pre-commit.yml)

![](figures/example_forecast.png)

SpaceCast is a repository for graph-based neural space weather forecasting. The code uses [PyTorch Lightning](https://lightning.ai/pytorch-lightning/) for modeling, and [Weights & Biases](https://wandb.ai/) for logging. The code is based on [Neural-LAM](https://github.com/mllam/neural-lam) and uses [MDP](https://github.com/mllam/mllam-data-prep) for data prep, which lowers the bar to adapt progress in limited area modeling for space weather.

The repository contains LAM versions of:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2024)](https://arxiv.org/abs/2406.04759).

## Dependencies

Use Python 3.10 / 3.11 and

- `torch==2.5.1`
- `pytorch-lightning==2.4.0`
- `torch_geometric==2.6.1`
- `mllam-data-prep==0.6.1`

Complete list of packages can be installed with `pip install -r requirements.txt`.

For linting further install `pre-commit install --install-hooks`. Then you can run `pre-commit run --all-files`.

## Quickstart

A small subset of the data is available for easy experimentation. Download with:
```
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deinal/spacecast-data-small",
    repo_type="dataset",
    local_dir="data_small"
)
```

Training can then be run immediately on the preprocessed data with readily available graphs.
```
python -m neural_lam.train_model \
  --config_path data_small/vlasiator_config.yaml \
  --model graph_efm \
  ...
```

## Data

The data is stored in [Zarr](https://zarr.dev) format on [Hugging Face](https://huggingface.co/datasets/deinal/spacecast-data).

It can be downloaded to a local `data` directory with:
```
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deinal/spacecast-data",
    repo_type="dataset",
    local_dir="data"
)
```

The folder will then follow the assumed structure of neural-lam:
```
data/
├── graph/                 - Directory containing graphs for training
├── run_1.zarr/            - Vlasiator run 1 with ρ = 0.5 cm⁻³ solar wind
├── run_2.zarr/            - Vlasiator run 2 with ρ = 1.0 cm⁻³ solar wind
├── run_3.zarr/            - Vlasiator run 3 with ρ = 1.5 cm⁻³ solar wind
├── run_4.zarr/            - Vlasiator run 4 with ρ = 2.0 cm⁻³ solar wind
├── static.zarr/           - Static features x, z, r coordinates
├── vlasiator_config.yaml  - Configuration file for neural-lam
├── vlasiator_run_1.yaml   - Configuration file for datastore 1, referred to from vlasiator_config.yaml
├── vlasiator_run_2.yaml   - Configuration file for datastore 2, referred to from vlasiator_config.yaml
├── vlasiator_run_3.yaml   - Configuration file for datastore 3, referred to from vlasiator_config.yaml
└── vlasiator_run_4.yaml   - Configuration file for datastore 4, referred to from vlasiator_config.yaml
```

Preprocess the runs with [mllam-data-prep](https://github.com/mllam/mllam-data-prep), run:
```
mllam_data_prep data/vlasiator_run_1.yaml
mllam_data_prep data/vlasiator_run_2.yaml
mllam_data_prep data/vlasiator_run_3.yaml
mllam_data_prep data/vlasiator_run_4.yaml
```
This produces training-ready zarr stores in the data directory.

Simple, multiscale, and hierarchical graphs are included already, but can be created using the following commands:
```
python -m neural_lam.create_graph --config_path data/vlasiator_config.yaml --name simple --levels 1 --coarsen-factor 5 --plot
python -m neural_lam.create_graph --config_path data/vlasiator_config.yaml --name multiscale --levels 3 --coarsen-factor 5 --plot
python -m neural_lam.create_graph --config_path data/vlasiator_config.yaml --name hierarchical --levels 3 --coarsen-factor 5 --hierarchical --plot
```

To plot the graphs and store as `.html` files run:
```
python -m neural_lam.plot_graph --datastore_config_path data/vlasiator_config.yaml --graph ...
```
with `--graph` as `simple`, `multiscale` or `hierarchcial` and `--save` specifies the name of the output file.

## Logging

If you'd like to login and use [W&B](https://wandb.ai/), run:
```
wandb login
```
If you prefer to just log things locally, run:
```
wandb off
```
See [docs](https://docs.wandb.ai/) for more details.

## Training

The first stage of a probabilistic model can be trained something like this (where in later stages you add `kl_beta` and `crps_weight`):

```
python -m neural_lam.train_model \
    --config_path data/vlasiator_config.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model graph_efm \
    --graph multiscale \
    --hidden_dim 64 \
    --processor_layers 4 \
    --ensemble_size 5 \
    --batch_size 1 \
    --lr 0.001 \
    --kl_beta 0 \
    --crps_weight 0 \
    --ar_steps_train 1 \
    --epochs 500 \
    --val_interval 50 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 2 3
```

Distributed data parallel training is supported. Specify number of nodes with the `--node` argument. For a full list of training options see `python neural_lam.train_model --help`.

## Evaluation

Inference uses the same script as training, with the same choice of parameters, and some to have an extra look at like `--eval test`, `--ar_steps_eval 30` and `--n_example_pred 1` to evaluate 30 second forecasts on the test set with 1 example forecast plotted.

```
python -m neural_lam.train_model \
  --config_path data/vlasiator_config.yaml \
  --model graph_efm \
  --graph hierarchical \
  --num_nodes 1 \
  --num_workers 2 \
  --batch_size 1 \
  --hidden_dim 64 \
  --processor_layers 2 \
  --ensemble_size 5 \
  --ar_steps_eval 30 \
  --precision bf16-mixed \
  --n_example_pred 1 \
  --eval test \
  --load ckpt_path
```
where a model checkpoint from a given path given to the `--load` in `.ckpt` format. Already trained model weights are available on [Zenodo](https://zenodo.org/records/16930055).

## Cite

ML dataset
```
@misc{vlasiator2025mldata,
  title={Vlasiator Dataset for Machine Learning Studies},
  author={Zaitsev, Ivan and Holmberg, Daniel and Alho, Markku and Bouri, Ioanna and Franssila, Fanni and Jeong, Haewon and Palmroth, Minna and Roos, Teemu},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/deinal/spacecast-data},
  doi={10.57967/hf/7027},
}
```

ML4PS paper
```
@inproceedings{holmberg2025graph,
    title={Graph-based Neural Space Weather Forecasting},
    author={Holmberg, Daniel and Zaitsev, Ivan and Alho, Markku and Bouri, Ioanna and Franssila, Fanni and Jeong, Haewon and Palmroth, Minna and Roos, Teemu},
    booktitle={NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences},
    year={2025}
}
```
This work is based on code using a single run dataloader at commit: https://github.com/fmihpc/spacecast/commit/937094079c1364ec484d3d1647e758f4a388ad97.

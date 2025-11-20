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
  --model graphcast \
  --graph simple \
  --epochs 10 \
  --lr 0.001 \
  --batch_size 4 \
  --hidden_dim 32 \
  --processor_layers 6 \
  --decode_dim 16 \
  --div_weight 10 \
  --ar_steps_train 2 \
  --ar_steps_eval 2
```

For more commands see:
```
python -m neural_lam.train_model --help
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
python -m neural_lam.create_graph \
  --config_path data/vlasiator_config.yaml \
  --name simple --levels 1 --coarsen-factor 5 --plot

python -m neural_lam.create_graph \
  --config_path data/vlasiator_config.yaml \
  --name multiscale --levels 3 --coarsen-factor 5 --plot

python -m neural_lam.create_graph \
  --config_path data/vlasiator_config.yaml \
  --name hierarchical --levels 3 --coarsen-factor 5 --hierarchical --plot
```

To plot the graphs and store as `.html` files run:
```
python -m neural_lam.plot_graph --datastore_config_path data/vlasiator_config.yaml --graph ...
```
with `--graph` as `simple`, `multiscale` or `hierarchcial` and `--save` specifies the name of the output file.

## Models

Pretrained models can be downloaded from [Hugging Face](https://huggingface.co/deinal/spacecast-models) using:
```
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deinal/spacecast-models",
    repo_type="model",
    local_dir="model_weights"
)
```
This also includes metrics for the models, and example forecasts for each run.

To reproduce results, run:
```
python -m neural_lam.plot_metrics \
  --metrics_dir model_weights/metrics \
  --forecasts_dir model_weights/forecasts \
  --output_dir model_weights/plots
```

Examples forecast animations are available online for [Run 1](https://vimeo.com/1138703695), [Run 2](https://vimeo.com/1138703709), [Run 3](https://vimeo.com/1138703719) and [Run 4](https://vimeo.com/1138703728).

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

For a full list of training options see `python neural_lam.train_model --help`.

The Graph-FM models were trained with commands like this:
```
python -m neural_lam.train_model \
  --config_path data/vlasiator_config.yaml \
  --model graphcast \
  --graph simple \
  --precision bf16-mixed \
  --epochs 250 \
  --scheduler_epochs 175 225 \
  --lr 0.001 \
  --batch_size 1 \
  --hidden_dim 256 \
  --processor_layers 12 \
  --decode_dim 128 \
  --ar_steps_train 4 \
  --div_weight 10 \
  --ar_steps_eval 4 \
  --num_sanity_val_steps 0 \
  --grad_checkpointing \
  --num_workers 4 \
  --num_nodes 4
```
The graph can be changed from `simple` to `multiscale` or `hierarchical`. In the case of the `hierarchical` graph change the `--model` from `graphcast` to `graph_fm`. Distributed data parallel training is supported, and the above script runs on 4 compute nodes. Gradient checkpointing is also turned on as training with many autoregressive steps increases memory consumption.

The probabilistic Graph-EFM model can be trained like this:
```
python -m neural_lam.train_model \
  --config_path data/vlasiator_config.yaml \
  --precision bf16-mixed \
  --model graph_efm \
   --graph multiscale \
   --hidden_dim 256 \
   --processor_layers 12 \
   --decode_dim 128 \
   --batch_size 1 \
   --lr 0.001 \
   --kl_beta 1 \
   --ar_steps_train 4 \
   --div_weight 1e8 \
   --crps_weight 1e6 \
   --epochs 250 \
   --scheduler_epochs 100 150 200 225
   --val_interval 5 \
   --ar_steps_eval 1 \
   --val_steps_to_log 1 \
   --num_sanity_val_steps 0 \
   --var_leads_val_plot '{"0":[1], "3":[1], "6":[1], "9":[1]}' \
   --grad_checkpointing \
   --num_workers 4 \
   --num_nodes 4
```
It is also possible to train without the `--scheduler_epochs`.  Sometimes it is more convenient to train each phase separately to tune loss weights for example. In this case manually train the model in phases with 1. `--kl_beta 0` off for autoencoder training, 2. then turn it on with `--kl_beta 1` for 1-step ELBO training, 3. increase `--ar_steps_train 4` to a suitable value, 4. turn on `--crps_weight 1e6` where you see decrease in the CRPS loss and increase in SSR, and 5. optionally apply `--div_weight 1e7` (which can be turned on earlier too).

## Evaluation

Inference uses similar scripts as training, and some evaluation specific flags like `--eval test`, `--ar_steps_eval 30` and `--n_example_pred 1` to evaluate 30 second forecasts on the test set with 1 example forecast plotted. Graph-FM can be evaluated using:
```
python -m neural_lam.train_model \
  --config_path data/vlasiator_config_4.yaml \
  --model graphcast \
  --graph simple \
  --precision bf16-mixed \
  --batch_size 1 \
  --hidden_dim 256 \
  --processor_layers 12 \
  --decode_dim 128 \
  --num_sanity_val_steps 0 \
  --num_workers 4 \
  --num_nodes 1 \
  --eval test \
  --ar_steps_eval 30 \
  --n_example_pred 0 \
  --load model_weights/graph_fm_simple.ckpt
```

Graph-EFM can be evaluated producing `--ensemble_size 5` as follows:
```
python -m neural_lam.train_model \
  --config_path data/vlasiator_config.yaml \
  --precision bf16-mixed \
  --model graph_efm \
  --graph simple \
  --hidden_dim 256 \
  --processor_layers 12 \
  --decode_dim 128 \
  --ensemble_size 5 \
  --batch_size 1 \
  --num_sanity_val_steps 0 \
  --num_workers 4 \
  --num_nodes 1 \
  --eval test \
  --ar_steps_eval 30 \
  --n_example_pred 0 \
  --load model_weights/graph_efm_simple.ckpt
```
where a model checkpoint from a given path given to the `--load` in `.ckpt` format. These examples use the pretrained models from [Hugging Face](https://huggingface.co/deinal/spacecast-models) stored in a `model_weights` directory.

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
The workshop paper is based on code using a single run dataloader at commit: [fmihpc/spacecast@ce3cd1](https://github.com/fmihpc/spacecast/tree/ce3cd1).

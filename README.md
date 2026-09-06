# ZSTNP

This is the official implementation of **Zero-Sum Set Transformer Neural Processes** (IEEE ICDM 2026).

The repository includes two groups of experiments:

- `regression/`: one-dimensional Gaussian process regression and image completion on CelebA, CIFAR-10, and EMNIST.
- `bayesian_optimization/`: one-dimensional and high-dimensional Bayesian optimization (BO) experiments.

The main ZSTNP configurations are located in the `configs/` directory of each experiment, while the model implementations are located in the corresponding `models/zstnp.py` and `models/zeros.py` files.

## Environment Setup

### Recommended Environment

The complete project dependencies are recorded in `environment.yml`. The reference environment uses:

- Python 3.7.2
- PyTorch 1.13.1
- torchvision 0.14.1
- BoTorch 0.5.1
- GPyTorch 1.6.0
- BayesO 0.4.3

```bash
conda env create --name ltnp --file environment.yml
conda activate ltnp
```

## Regression Experiments

### One-Dimensional GP Regression

Train ZSTNP:

```bash
python gp.py --mode train --model zstnp --expid default-zstnp --device cuda:0
```

Evaluate a checkpoint:

```bash
python gp.py --mode eval --model zstnp --expid default-zstnp --device cuda:0 --eval_kernel rbf
```

`--eval_kernel` supports `rbf`, `matern`, and `periodic`. Training and evaluation must use the same `--expid`.

### CIFAR-10 Image Completion

Download and preprocess CIFAR-10 before the first run:

```bash
python data/cifar.py --resolution 32
```

Train and evaluate:

```bash
python cifar.py --mode train --model zstnp --expid default-zstnp --resolution 32 --device cuda:0
python cifar.py --mode eval  --model zstnp --expid default-zstnp --resolution 32 --device cuda:0
```

### EMNIST Image Completion

EMNIST is downloaded automatically on the first run. Use the following commands to train and evaluate:

```bash
python emnist.py --mode train --model zstnp --expid default-zstnp --device cuda:0
python emnist.py --mode eval  --model zstnp --expid default-zstnp --device cuda:0
```

### CelebA Image Completion

Download the CelebA files and arrange them as follows:

```text
regression/datasets/celeba/
├── img_align_celeba/
├── list_eval_partition.txt
└── identity_CelebA.txt
```

Preprocess the dataset:

```bash
mkdir -p datasets/celeba32
python data/celeba.py --resolution 32
```

Train and evaluate:

```bash
python celeba.py --mode train --model zstnp --expid default-zstnp --resolution 32 --device cuda:0
python celeba.py --mode eval  --model zstnp --expid default-zstnp --resolution 32 --device cuda:0
```

## Bayesian Optimization Experiments

### One-Dimensional BO

```bash
python 1d_gp.py --mode train --model zstnp --expid default
```

```bash
python 1d_bo.py --bo_mode models --bo_kernel rbf --model zstnp --expid default
```

### High-Dimensional BO

```bash
python highdim_gp.py --mode train --dimension 2 --model zstnp --min_num_points 30 --max_num_points 128 --num_steps 100000
```

```bash
python highdim_gp.py --mode eval --dimension 2 --model zstnp --min_num_points 30 --max_num_points 128 --num_steps 100000
```

The current implementation supports the following combinations of dimensions and objective functions:

| Dimension | Objective functions |
|---|---|
| Any positive integer | `ackley`, `cosine`, `rastrigin` |
| 2D | `dropwave`, `eggholder`, `goldsteinprice`, `michalewicz`, `schaffer`, `griewank` |
| 3D | `hartmann` |

To view the complete argument list for any script:

```bash
python gp.py --help
python 1d_bo.py --help
python highdim_gp.py --help
python highdim_bo.py --help
```

## Outputs and Checkpoints

- Regression outputs: `regression/results/`
- Automatically generated regression evaluation sets: `regression/evalsets/`
- BO outputs: `bayesian_optimization/results/`
- High-dimensional GP training data: `bayesian_optimization/datasets/`
- Automatically generated high-dimensional GP evaluation sets: `bayesian_optimization/evalsets/`

Training checkpoints are named `ckpt.tar` by default. Before running evaluation or BO, make sure this file has been generated in the result directory corresponding to the training command.

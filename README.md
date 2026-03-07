# DA6401 Assignment 1 - Neural Network From Scratch

**GitHub Repo Link:** https://github.com/NagasaiManda/DA6001_PA-1-/tree/main

**Report Link:**  https://api.wandb.ai/links/ee23b042-indian-institute-of-technology-madras/z1ocwvw3

## Project Summary
This project implements a fully-connected neural network from scratch using NumPy for image classification on `MNIST` and `Fashion-MNIST`.

Implemented modules include:
- Dense layer (`fc`) with manual backward pass
- Activations: `ReLU`, `Sigmoid`, `Tanh`
- Losses: `CrossEntropyx`, `MSE`
- Optimizers: `SGD`, `Momentum`, `RMSprop`, `NAG`
- Training and inference scripts

## Repository Structure
- `src/ann/`: core neural-network implementation
- `src/utils/data_loader.py`: dataset loading and preprocessing
- `src/train.py`: training entry point
- `src/inference.py`: evaluation/inference entry point
- `models/`: saved model checkpoints
- `notebooks/`: exploratory/experiment notebooks

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train


```bash
python src/train.py \
  -d mnist \
  -e 10 \
  -b 32 \
  -o rmsprop \
  -lr 0.001 \
  -nhl 2 \
  -sz 128 64 \
  -a relu \
  -l cross_entropy \
  -w_i xavier \
  -mp models/model.npy
```

## Inference

```bash
python src/inference.py \
  -d mnist \
  -mp models/model.npy \
  -nhl 2 \
  -sz 128 64 \
  -a relu \
  -l cross_entropy
```


## Notes
- Labels are one-hot encoded in the data loader.
- Input images are flattened to 784-dimensional vectors and normalized to `[0, 1]`.
- Ensure model architecture flags at inference match those used during training.

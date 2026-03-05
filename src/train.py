"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from html import parser
from ann.neural_network import NeuralNetwork
import numpy as np

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist"],
        required=True,
        help="Choose dataset: mnist or fashion_mnist"
    )

    # Epochs
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )

    # Batch size
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        required=True,
        help="Mini-batch size"
    )

    # Loss function
    parser.add_argument(
        "-l", "--loss",
        type=str,
        choices=["mean_squared_error", "cross_entropy"],
        required=True,
        help="Loss function"
    )

    # Optimizer
    parser.add_argument(
        "-o", "--optimizer",
        type=str,
        choices=["sgd", "momentum", "nag", "rmsprop"],
        required=True,
        help="Optimizer"
    )

    # Learning rate
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        required=True,
        help="Initial learning rate"
    )

    # Weight decay
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for L2 regularization"
    )

    # Number of hidden layers
    parser.add_argument(
        "-nhl", "--num_layers",
        type=int,
        required=True,
        help="Number of hidden layers"
    )

    # Hidden layer sizes
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int,
        nargs="+",
        required=True,
        help="Number of neurons in each hidden layer"
    )

    # Activation
    parser.add_argument(
        "-a", "--activation",
        type=str,
        choices=["sigmoid", "tanh", "relu"],
        required=True,
        help="Activation function for hidden layers"
    )

    # Weight initialization
    parser.add_argument(
        "-wi", "--weight_init",
        type=str,
        choices=["random", "xavier"],
        required=True,
        help="Weight initialization method"
    )

    # W&B project
    parser.add_argument(
        "-wp", "--wandb_project",
        type=str,
        required=False,
        help="Weights & Biases project ID"
    )

    parser.add_argument(
        "-mp", "--model_path",
        type=str,
        default="models/model.npy",
        help="Relative path to save trained model"
    )




    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    model = NeuralNetwork(args)
    model.train()
    weights = model.get_weights()
    np.save(args.model_path, weights)

    
    print("Training complete!")


if __name__ == '__main__':
    main()

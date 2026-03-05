"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
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
        "-w_i", "--weight_init",
        type=str,
        choices=["random", "xavier"],
        required=True,
        help="Weight initialization method"
    )

    # W&B project
    parser.add_argument(
        "-w_p", "--wandb_project",
        type=str,
        default="default_project",
        help="Weights & Biases project ID"
    )

    parser.add_argument(
        "-mp", "--model_path",
        type=str,
        default="models/model.npy",
        help="Relative path to save trained model"
    )
    return parser.parse_args()



def load_model(model_path):
    """
    Load trained model from disk.
    """
    weights = np.load(model_path, allow_pickle=True).item()  
    return weights




def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    y_pred = model.forward(X_test)
    loss = model.loss_fn.forward(y_pred, y_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    return {
        "logits": y_pred,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    dataset = load_data(args.dataset)
    X_test, y_test = dataset[1]  
    model = NeuralNetwork(args)  
    model.set_weights(load_model(args.model_path)) 
    results = evaluate_model(model, X_test, y_test)
    print("Evaluation complete!")
    print(results)
    return results

if __name__ == '__main__':
    main()

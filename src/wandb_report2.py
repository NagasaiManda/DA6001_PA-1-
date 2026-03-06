import wandb
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from utils.data_loader import load_data


###########################################
# Sweep Configuration
###########################################

sweep_config = {
    "method": "random",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {

        "learning_rate": {
            "values": [1e-4, 5e-4, 1e-3, 5e-3]
        },

        "batch_size": {
            "values": [32, 64, 128]
        },

        "optimizer": {
            "values": ["sgd", "momentum", "nag", "rmsprop"]
        },

        "activation": {
            "values": ["relu", "tanh", "sigmoid"]
        },

        "num_layers": {
            "values": [1,2,3,4,5,6]
        },

        "layer1_size": {"values": [32,64,128]},
        "layer2_size": {"values": [32,64,128]},
        "layer3_size": {"values": [32,64,128]},
        "layer4_size": {"values": [32,64,128]},
        "layer5_size": {"values": [32,64,128]},
        "layer6_size": {"values": [32,64,128]},

        "weight_decay": {
            "values": [0,1e-5,1e-4,1e-3]
        },

        "epochs": {
            "value": 10
        }
    }
}


###########################################
# Utility: Build Hidden Layer List
###########################################

def build_hidden_sizes(config):

    sizes = [
        config.layer1_size,
        config.layer2_size,
        config.layer3_size,
        config.layer4_size,
        config.layer5_size,
        config.layer6_size
    ]

    return sizes[:config.num_layers]


###########################################
# Sweep Training Function
###########################################

def train_sweep():

    run = wandb.init()

    config = wandb.config

    hidden_sizes = build_hidden_sizes(config)

    cli_args = SimpleNamespace(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        activation=config.activation,
        hidden_size=hidden_sizes,
        num_layers=config.num_layers,
        weight_decay=config.weight_decay,
        dataset="mnist",
        loss="cross_entropy",
        weight_init="xavier",
        wandb_project="nn-hyperparam-sweep",
        model_path="best_model.npy"
    )

    model = NeuralNetwork(cli_args)

    (X_train, y_train), (X_test, y_test) = load_data("mnist")

    ####################################
    # TRAIN MODEL
    ####################################

    val_acc_history = model.train()

    ####################################
    # Log per-epoch metrics
    ####################################

    for epoch, val_acc in enumerate(val_acc_history):

        wandb.log({
            "epoch": epoch,
            "val_accuracy": val_acc
        })

    ####################################
    # Final Evaluation
    ####################################

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    wandb.log({

        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],

        "train_loss": train_metrics["loss"],
        "test_loss": test_metrics["loss"],

        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],

        "architecture": str(hidden_sizes)
    })

    run.finish()


###########################################
# Run Sweep
###########################################

if __name__ == "__main__":
    PROJECT_NAME = 'DA6401_Assignment_1'
    sweep_id = wandb.sweep(
        sweep_config,
        project=PROJECT_NAME
    )

    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=100
    )
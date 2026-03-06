"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import fc
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import MSE, CrossEntropyWithSoftmax
from .optimizers import SGD, Momentum, RMSprop, NAG
from utils.data_loader import load_data
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.epochs = getattr(cli_args, "epochs", 10)
        self.batch_size = getattr(cli_args, "batch_size", 62)
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)

        self.wandb_project = getattr(cli_args, "wandb_project", None)
        self.model_save_path = getattr(cli_args, "model_path", "src/best_model.npy")

        self.learning_rate = getattr(cli_args, "learning_rate", 0.001)
        self.optimizer_name = getattr(cli_args, "optimizer", "rmsprop")
        self.dataset = getattr(cli_args, "dataset", "mnist")
        self.loss = getattr(cli_args, "loss", "cross_entropy")

        self.num_layers = getattr(cli_args, "num_layers", 2)
        self.hidden_size = getattr(cli_args, "hidden_size", [128, 64])
        self.activation = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")



        # self.epochs = cli_args.epochs
        # self.batch_size = cli_args.batch_size
        # self.weight_decay = cli_args.weight_decay
        # self.wandb_project = cli_args.wandb_project
        # self.model_save_path = cli_args.model_path

        # self.learning_rate = cli_args.learning_rate
        # self.optimizer_name = cli_args.optimizer
        # self.dataset = cli_args.dataset
        # self.loss = cli_args.loss
        # self.num_layers = cli_args.num_layers
        # self.hidden_size = cli_args.hidden_size
        # self.activation = cli_args.activation
        # self.weight_init = cli_args.weight_init

        self.layers = []
        input_dim = 784
        output_dim = 10

        prev_dim = input_dim
        weight_init = 0 if self.weight_init == "random" else 1
        for i in range(self.num_layers):
            self.layers.append(
                fc(prev_dim, self.hidden_size[i], weight_init)
            )

            if self.activation == "relu":
                self.layers.append(ReLU())
            elif self.activation == "sigmoid":
                self.layers.append(Sigmoid())
            elif self.activation == "tanh":
                self.layers.append(Tanh())

            prev_dim = self.hidden_size[i]

        self.layers.append(fc(prev_dim, output_dim, weight_init))

        if self.loss == "cross_entropy":
            self.loss_fn = CrossEntropyWithSoftmax()
        else:
            self.loss_fn = MSE()

        self.data = load_data(self.dataset)     #(x_train, y_train), (x_test, y_test))
        self.x_train, self.y_train = self.data[0]   


        self.parameters = {
            "params": [layer.W for layer in self.layers if isinstance(layer, fc)] + [layer.b for layer in self.layers if isinstance(layer, fc)],
            "grads": [layer.grad_W for layer in self.layers if isinstance(layer, fc)] + [layer.grad_b for layer in self.layers if isinstance(layer, fc)]
        }
        if self.optimizer_name == "sgd":
            self.optimizer = SGD(self.parameters, lr=self.learning_rate)
        elif self.optimizer_name == "momentum":
            self.optimizer = Momentum(self.parameters, lr=self.learning_rate)
        elif self.optimizer_name == "nag":
            self.optimizer = NAG(self.parameters, lr=self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            self.optimizer = RMSprop(self.parameters, lr=self.learning_rate)

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, y_true, logits):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """

        self.loss_fn.forward(logits, y_true)
        for layer in self.layers:
            if isinstance(layer, fc):
                layer.zero_grad()
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        delta = self.loss_fn.backward(y_true, logits)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            if isinstance(layer, fc):
                grad_W_list.append(layer.grad_W.copy())
                grad_b_list.append(layer.grad_b.copy())

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        self.grad_W = self.grad_W[::-1]
        self.grad_b = self.grad_b[::-1]
        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step()
        for layer in self.layers:
            if isinstance(layer, fc):
                layer.zero_grad()

    def train(self):
        X_train = self.x_train
        y_train = self.y_train
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        epochs = self.epochs
        batch_size = self.batch_size
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            running_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if self.optimizer_name == "nag":
                    original_params = [p.copy() for p in self.parameters["params"]]

                    for i, p in enumerate(self.parameters["params"]):
                        p -= self.optimizer.gamma * self.optimizer.v[i]

                logits = self.forward(X_batch)
                loss = self.loss_fn.forward(logits, y_batch)

                running_loss += loss
                num_batches += 1

                self.backward(y_batch, logits)

                if self.optimizer_name == "nag":
                    for i, p in enumerate(self.parameters["params"]):
                        p[:] = original_params[i]

                self.update_weights()

            
            eval = self.evaluate(X_val, y_val)

            print(f"Epoch {epoch+1}/{epochs} completed.")
            print(f"Average Loss: {running_loss/num_batches:.4f}, Validation Loss: {eval['loss']:.4f}")

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss = self.loss_fn.forward(logits, y)
        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)

        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        return {
            "logits": logits,
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def get_weights(self):
        d = {}
        idx = 0
        for layer in self.layers:
            if isinstance(layer, fc):
                d[f"W{idx}"] = layer.W.copy()
                d[f"b{idx}"] = layer.b.copy()
                idx += 1
        return d

    def set_weights(self, weight_dict):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, fc):
                w_key = f"W{idx}"
                b_key = f"b{idx}"

                if w_key in weight_dict:
                    layer.W = weight_dict[w_key].copy()
                if b_key in weight_dict:
                    layer.b = weight_dict[b_key].copy()

                idx += 1



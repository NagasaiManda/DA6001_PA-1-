"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np 

# class CrossEntropy:
#     def forward(self, y_preds, y_true):
#         self.y_preds = y_preds
#         self.y_true = y_true
#         return -np.mean(np.sum(y_true * np.log(self.y_preds), axis=1))

#     def backward(self):
#         B = self.y_preds.shape[0]
#         return -(self.y_true / self.y_preds) / B





def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

class MSE:
    def forward(self, logits, y_true):
        self.y_pred = logits
        self.y_true = y_true
        loss = np.mean((self.y_pred - y_true) ** 2)
        return loss

    def backward(self):
        grad = 2.0 * (self.y_pred - self.y_true) / (self.y_true.shape[1] *self.y_true.shape[0])
        dot = np.sum(grad * self.y_pred, axis=1, keepdims=True)
        return self.y_pred * (grad - dot)


class CrossEntropyWithSoftmax:
    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - logsumexp
        self.probabilities = np.exp(log_probs)
        self.y_true = y_true
        loss = -np.sum(y_true * log_probs) / logits.shape[0]
        return loss

    def backward(self):
        return (self.probabilities - self.y_true) / self.y_true.shape[0]



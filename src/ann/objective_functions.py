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


def _ensure_2d(arr):
    arr = np.asarray(arr)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)
    return arr, was_1d


class MSE:
    def forward(self, logits, y_true):
        logits, _ = _ensure_2d(logits)
        y_true, _ = _ensure_2d(y_true)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        self.y_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_true = y_true
        return np.mean((self.y_preds - self.y_true)**2)

    __call__ = forward
    def backward(self, y_true, logits):
        logits, was_1d = _ensure_2d(logits)
        y_true, _ = _ensure_2d(y_true)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        y_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        grad = 2 * (y_preds - y_true) / y_preds.size
        return grad[0] if was_1d else grad



class CrossEntropyWithSoftmax:
    def forward(self, logits, y_true):
        logits, _ = _ensure_2d(logits)
        y_true, _ = _ensure_2d(y_true)
        self.y_true = y_true
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        self.y_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Clip probabilities to avoid log(0) and resulting NaNs/Infs.
        clipped_preds = np.clip(self.y_preds, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(clipped_preds), axis=1))

    __call__ = forward
    def backward(self, y_true, logits):
        logits, was_1d = _ensure_2d(logits)
        y_true, _ = _ensure_2d(y_true)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        y_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        grad = (y_preds - y_true) / y_preds.shape[0]
        return grad[0] if was_1d else grad

    

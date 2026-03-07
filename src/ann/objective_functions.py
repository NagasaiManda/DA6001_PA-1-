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


class MSE:
    def forward(self, y_preds, y_true):
        self.y_preds, self.y_true = y_preds, y_true
        return np.mean((y_preds - y_true)**2)

    __call__ = forward
    def backward(self, y_true=None, y_preds=None):
        return 2 * (self.y_preds - self.y_true) / self.y_preds.size



class CrossEntropyWithSoftmax:
    def forward(self, logits, y_true):
        self.y_true = y_true
        logits_shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shift)
        self.y_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        log_probs = logits_shift - logsumexp
        return -np.mean(np.sum(y_true * log_probs, axis=1))

    __call__ = forward
    def backward(self, y_true=None, logits=None):
        return (self.y_preds - self.y_true) / self.y_preds.shape[0]

    

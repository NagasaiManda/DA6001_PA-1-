"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class fc:
    def __init__(self, in_dims, out_dims, weight_init=0):
        self.in_dims = in_dims
        self.out_dims = out_dims
        if weight_init == 0:
            self.W = 0.01 * np.random.random((out_dims, in_dims))
        else:
            std = np.sqrt(2 / (self.in_dims + self.out_dims))
            self.W = np.random.normal(0, std, (out_dims, in_dims))

        self.b = np.zeros((1, out_dims))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.x = x
        return x @ self.W.T + self.b

    __call__ = forward

    def backward(self, delta):
        self.grad_W += delta.T @ self.x
        self.grad_b += np.sum(delta, axis=0, keepdims=True)
        return delta @ self.W

    def zero_grad(self):
        self.grad_W[...] = 0
        self.grad_b[...] = 0


    


"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for param, grad in zip(self.parameters, self.grads):
            param -= self.lr * (grad + self.weight_decay * param)
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0



class Momentum:
    def __init__(self, parameters, gamma=0.9, lr=0.01, weight_decay=0.0):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.v = [np.zeros_like(p) for p in self.parameters]

    def step(self):
        for i in range(len(self.parameters)):
            grad_reg = self.grads[i] + self.weight_decay * self.parameters[i]
            self.v[i] = self.gamma * self.v[i] + self.lr * grad_reg
            self.parameters[i] -= self.v[i]
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0



class RMSprop:
    def __init__(self, parameters, beta=0.9, lr=0.01, weight_decay=0.0):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v = [np.zeros_like(p) for p in self.parameters]

    def step(self):
        for i in range(len(self.parameters)):
            grad_reg = self.grads[i] + self.weight_decay * self.parameters[i]
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad_reg**2)
            self.parameters[i] -= (self.lr / (np.sqrt(self.v[i]) + 1e-8)) * grad_reg
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0



class NAG:
    def __init__(self, parameters, gamma=0.9, lr=0.01, weight_decay=0.0):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.v = [np.zeros_like(p) for p in self.parameters]

    def step(self):
        for i in range(len(self.parameters)):
            grad_reg = self.grads[i] + self.weight_decay * self.parameters[i]
            self.v[i] = self.gamma * self.v[i] + self.lr * grad_reg
            self.parameters[i] -= (self.gamma * self.v[i] + self.lr * grad_reg)
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0

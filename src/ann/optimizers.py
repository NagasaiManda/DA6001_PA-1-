"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr

    def step(self):
        for param, grad in zip(self.parameters, self.grads):
            param -= self.lr*grad
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0

class Momentum:
    def __init__(self,parameters, gamma=0.9, lr=0.01):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.gamma = gamma
        self.v = []
        for grad in self.grads:
            self.v.append(np.zeros_like(grad))

    def step(self):
        for v_i,grad in zip(self.v,self.grads):
            v_i[...] = self.gamma*v_i + self.lr*grad
        for param, v_i in zip(self.parameters, self.v):
            param -= v_i
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0

class RMSprop:
    def __init__(self,parameters, beta=0.9, lr=0.01):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.beta = beta
        self.v = []
        for grad in self.grads:
            self.v.append(np.zeros_like(grad))

    def step(self):
        for param, v_i, grad in zip(self.parameters, self.v, self.grads):
            v_i[...] = self.beta*v_i + (1-self.beta)*(grad*grad)
            param -= (self.lr/(np.sqrt(v_i)+ 1e-8)) * grad
    
    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0



class NAG:
    def __init__(self, parameters, gamma=0.9, lr=0.01):
        self.parameters = parameters["params"]
        self.grads = parameters["grads"]
        self.lr = lr
        self.gamma = gamma
        self.v = [np.zeros_like(p) for p in self.parameters]

    def step(self):
        for i in range(len(self.parameters)):
            self.v[i] = self.gamma * self.v[i] + self.lr * self.grads[i]
            self.parameters[i] -= self.v[i]


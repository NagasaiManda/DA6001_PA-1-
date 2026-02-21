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
            self.weights = 0.01*np.random.random((out_dims, in_dims))
        else:
            std = np.sqrt(2/(self.in_dims+self.out_dims))
            self.weights = np.random.normal(0,std, (out_dims, in_dims))
        self.biases = np.zeros(out_dims)
        self.grad_W = np.zeros_like(self.weights)   
        self.grad_b = np.zeros_like(self.biases)    # Treat it as a vector?
        
    def forward(self,x):
        self.x = x
        return self.weights @ x + self.biases
    
    __call__ = forward
    
    def backward(self,delta):
        self.grad_W += np.outer(delta, self.x)
        self.grad_b += delta
        return self.weights.T @ delta
    
    def zero_grad(self):
        self.grad_W[...] = 0
        self.grad_b[...] = 0
    



    


"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

class ReLU:
    def forward(self,x):
        self.x = x
        return np.maximum(0,x)
    
    __call__ = forward

    def backward(self, delta):
        return delta*(self.x > 0)
    
class Sigmoid:
    def forward(self,x):
        self.x = x
        # Numerically stable sigmoid to avoid overflow warnings for large |x|.
        self.res = np.empty_like(x)
        pos_mask = x >= 0
        self.res[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[~pos_mask])
        self.res[~pos_mask] = exp_x / (1.0 + exp_x)
        return self.res
    
    __call__ = forward
    
    def backward(self,delta):
        return delta * self.res * (1-self.res)
    
class Tanh:
    def forward(self,x):
        self.x = x
        self.res = np.tanh(x) 
        return self.res
    
    __call__ = forward

    def backward(self, delta):
        return delta * (1-np.square(self.res))
    
# class Softmax:
#     def forward(self, x):
#         x = x - np.max(x, axis=1, keepdims=True)  
#         res = np.exp(x)
#         self.res = res / np.sum(res, axis=1, keepdims=True)
#         return self.res

#     __call__ = forward

#     def backward(self, delta):
#         dot = np.sum(self.res * delta, axis=1, keepdims=True)
#         return self.res * (delta - dot)

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
        self.res = 1/(1+np.exp(-x))
        return self.res
    
    __call__ = forward
    
    def backward(self,delta):
        return delta * self.res * (1-self.res)
    
class Tanh:
    def forward(self,x):
        self.x = x
        

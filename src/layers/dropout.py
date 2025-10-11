import numpy as np
from src.utils.parameters import Parameters


class Dropout:
    def __init__(self, rate = 0.2):
        self.rate = rate

    def forward(self, X):
        self.p = np.random.binomial(1,1-self.rate, size = X.shape)
        return X*self.p
    
    def backward(self, error):
        return error * self.p
    
    def parameters(self):
        return []


    
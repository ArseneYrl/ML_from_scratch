import numpy as np
from .base import Optimizer

class SGD_momentum(Optimizer):
    def __init__(self, parameters, lr, beta1):
        super().__init__(parameters, lr)
        self.beta1=beta1
        self.mt = [np.zeros_like(w.data) for w in self.parameters]  


    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.mt[i] = self.beta1 * self.mt[i] + param.grad
                param.data -= self.lr * self.mt[i]
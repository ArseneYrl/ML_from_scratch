import numpy as np
from .base import Optimizer

class RMSprop(Optimizer):
    #RMSprop optimizer
    def __init__(self, parameters, lr, gamma):
        super().__init__(parameters, lr)
        self.gamma=gamma
        self.gt = [np.zeros_like(w.data) for w in self.parameters]  


    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.gt[i] = self.gamma * self.gt[i] + (1-self.gamma) * (param.grad**2)
                param.data -= self.lr * param.grad/(np.sqrt(self.gt[i])+1e-8)
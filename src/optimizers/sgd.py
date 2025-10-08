import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)

    def step(self):
        for param in self.parameters:
            if param is not None:
                param.data -= self.lr * param.grad

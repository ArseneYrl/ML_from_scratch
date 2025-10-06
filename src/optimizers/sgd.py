import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)


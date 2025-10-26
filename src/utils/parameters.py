import numpy as np

#Class to store both the data and the grad of the data in the optimizer/backprop
class Parameters:
    def __init__(self, data):
        self.data = data
        self.grad = None
import numpy as np
from src.layers.linear import Linear
from src.layers.activation import ReLU, Tanh


class MLP:
    def __init__(self):
        self.layers=[]

    def add_layer(self,layer):
        self.layers.append(layer)

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self,err):
        for layer in reversed(self.layers):
            err = layer.backward(err)

    def parameters(self):
        return self.layers


import numpy as np

class Cnn:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self):
        pass

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer,'parameters'):
                params.extend(layer.parameters())
        return params
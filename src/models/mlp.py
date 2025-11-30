import numpy as np
from src.layers.linear import Linear
from src.layers.activation import ReLU, Tanh


class MLP:
    def __init__(self):
        self.layers=[]
        
    def add_layer(self,layer):
        #Add a layer to the network
        self.layers.append(layer)

    def feedforward(self, X):
        #Feedforward through all the layers
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self,err):
        #Backprop through all the layers
        for layer in reversed(self.layers):
            err = layer.backward(err)   

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer,'parameters'):
                params.extend(layer.parameters())
        return params


import numpy as np
from src.utils.parameters import Parameters

class Linear:
    #Fully connected layer
    def __init__(self,input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        std = np.sqrt(2.0 / self.input_size)

        self.W = Parameters(np.random.normal(0, std, size=(self.input_size, self.output_size))) #Initialization of the weights
        self.b = Parameters(np.zeros((1, self.output_size))) #Initialization of the biases


    def forward(self,X):
        #Feedforward through the layer
        self.X = X
        return X @ self.W.data + self.b.data
    
    def backward(self, error):
        #Backward propagation
        batch_size = error.shape[0]

        self.W.grad = self.X.T @ error / batch_size
        self.b.grad = np.sum(error, axis=0, keepdims=True) / batch_size

        return error @ self.W.data.T
    
    def parameters(self):
        #Parameters
        return [self.W, self.b]
    
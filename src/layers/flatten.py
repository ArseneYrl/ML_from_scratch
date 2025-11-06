import numpy as np

class Flatten:
    #Flatten layer
    def __init__(self):
        pass

    def forward(self,X):
        self.X=X
        return np.ndarray.flatten(X)
    
    def backward(self,error):
        return error.reshape(self.X.shape)

    def parameters(self):
        return []
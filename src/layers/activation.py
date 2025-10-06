import numpy as np

class ReLU: #ReLU activation
    def forward(self, z):
        self.z = z
        return np.maximum(0,z) 
    
    def backward(self,error):
        dReLU = (self.z > 0).astype(float)
        return error * dReLU

class Tanh: #tanh activation
    def forward(self, z):
        self.z = z
        return np.tanh(z)
    
    def backward(self, error):
        dTanh = 1 - np.tanh(self.z)**2
        return error * dTanh
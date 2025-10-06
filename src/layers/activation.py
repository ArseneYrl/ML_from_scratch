import numpy as np

class ReLU:
    def forward(self, z):
        self.z = z
        return np.maximum(0,z) #Return ReLU(z), activation
    
    def backward(self,error):
        dReLU = (self.z > 0).astype(float)
        return error * dReLU

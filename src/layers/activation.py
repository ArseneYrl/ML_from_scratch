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
    
class Sigmoid: #Sigmoid activaion
    def forward(self, z):
        self.z = z
        return 1/(1-np.exp(-z))
    
    def backward(self, error):
        exp = np.exp(-self.z- np.max(error, axis=1, keepdims=True))
        dSig = -exp/((1-exp)**2)
        return error * dSig
    
class Leaky_ReLU: #Leaky_ReLU activation
    def forward(self, z):
        self.z = z
        self.cond = (self.z > 0)
        return np.where(self.cond,0.1*self.z, self.z)
    
    def backward(self, error):
        dLRLU = np.where(self.cond,0.1,1)
        return error * dLRLU
    
class ELU: #ELU activation
    def forward(self, z):
        self.z = z
        self.cond = (self.z > 0)
        self.exp = np.exp(self.z-np.max(z, axis=1, keepdims=True))
        return np.where(self.cond, self.exp-1, self.z)
    
    def backward(self, error):
        dELU = np.where(self.cond, self.exp, 1)
        return error * dELU



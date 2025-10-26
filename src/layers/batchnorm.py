import numpy as np

class Batch_normalization:
    def __init__(self):
        pass

    def forward(self, X_batch):
        mu = np.mean(X_batch)
        sig = np.var(X_batch)
        X_hat = (X_batch-mu)/np.sqrt(sig + 10e-8)
        
    
    def backward(self, error):
        return error * self.p
    
    def parameters(self):
        return []
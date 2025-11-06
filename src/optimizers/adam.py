import numpy as np
from .base import Optimizer

class ADAM(Optimizer):
    #ADAM Optimizer
    
    def __init__(self, parameters, lr, b1, b2):
        super().__init__(parameters, lr)
        self.gt = [np.zeros_like(w.data) for w in self.parameters] 
        self.mt = [np.zeros_like(w.data) for w in self.parameters]  

        self.epoch = 1
        
        self.b1 = b1
        self.b2 = b2


    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.mt[i] = self.b1 * self.mt[i] + (1-self.b1)*param.grad
                self.gt[i] = self.b2 * self.gt[i] + (1-self.b2) * (param.grad**2)

                mhat = self.mt[i]/(1-self.b1**self.epoch)
                ghat = self.gt[i]/(1-self.b2**self.epoch)

                param.data -= self.lr * mhat/(np.sqrt(ghat)+1e-8)
                self.epoch +=1
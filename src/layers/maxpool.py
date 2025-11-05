import numpy as np
from src.utils.parameters import Parameters
from src.utils.center import Patches

class MaxPool:
    def __init__(self, F=3, S=1):
        self.F = F
        self.S = S

    def forward(self, X):
        self.patches = Patches(self.F, self.S)
        self.centers=self.patches.separate(X)
        self.D,self.H,self.W = X.shape
        D1=self.D
        H1=(self.H-self.F)//self.S+1
        W1=(self.W-self.F)//self.S+1
        output = np.zeros([D1,H1,W1])
        self.switches = np.zeros([D1,H1,W1])
        
        for k in range(D1):
            count=0
            for i in range(H1):
                for j in range(W1):
                    output[k,i,j]=np.max(self.centers[count][k])
                    self.switches[k,i,j]=np.argmax(self.centers[count][k])
                    count+=1
        return output
    
    def backward(self, error):
        count=0
        start_grads = np.zeros_like(self.centers,dtype=float)
        for k in range(self.switches.shape[0]):
            count=0
            for i in range(self.switches.shape[1]):
                for j in range(self.switches.shape[2]):
                    max_pos = int(self.switches[k,i,j])
                    start_grads[count].flat[max_pos]=error[k,i,j]
                    count+=1

        return self.patches.reconstruct_grad(start_grads)

    def parameters(self):
        return []
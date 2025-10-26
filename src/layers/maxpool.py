import numpy as np
from src.utils.parameters import Parameters
from src.utils.center import center

class MaxPool:
    def __init__(self, F=3, S=1):
        self.F = F
        self.S = S

    def forward(self, X):
        centers = center(X, self.F, self.S)
        D1,H1,W1 = X.shape
        D1=self.D
        H1=(self.H-self.F)//self.S+1
        W1=(self.W-self.F)//self.S+1
        output = np.zeros([D1,H1,W1])
        
        for k in range(D1):
            count=0
            for i in range(H1):
                for j in range(W1):
                    output[k,i,j]=np.max(centers[count][k])
                    count+=1
        return output
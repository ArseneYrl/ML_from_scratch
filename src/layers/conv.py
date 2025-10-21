import numpy as np
from src.utils.parameters import Parameters

class Conv:
    def __init__(self, D, W, H, K, F = 3, S = 1, P=0):
        self.W = W
        self.H = H
        self.D = D
        self.F = F
        self.S = S
        self.K = K
        self.P = P
        self.filters = Parameters(np.random.randn(self.K, self.D, self.F, self.F))
        self.b = Parameters(np.random.randn(self.K))

    def center(self, Input):
        #Input will be of shape (W, H, D)
        centers = []
        rad = int((self.F-1)/2)
        for i in range(rad-1,Input.shape[1]-rad-1,self.S):
            for j in range(rad-1,Input.shape[2]-rad-1,self.S):
                C=Input[::,i:i+self.F,j:j+self.F]
                centers.append(C)
        return centers
    

    def forward(self,inputs):
        centers = self.center(self.zero_padding(inputs))
        H1 = int((self.H+2*self.P-self.F)/self.S +1)
        W1 = int((self.W+2*self.P-self.F)/self.S +1)
        D1 = self.K
        output = np.zeros([D1, H1, W1])
        for k in range(self.filters.data.shape[0]):
            count = 0
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    output[k,i,j] = np.sum(self.filters.data[k]*centers[count])+self.b.data[k]
                    count+=1
        return output


    def zero_padding(self,input):
        padding = np.zeros([self.D,self.H+2*self.P, self.W+2*self.P])
        padding[::,self.P:self.H+1, self.P:self.W+1]=input
        return padding
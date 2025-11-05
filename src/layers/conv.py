import numpy as np
from src.utils.parameters import Parameters
from src.utils.patches import Patches
class Conv:
    def __init__(self, D, K, F = 3, S = 1, P=0):
        self.D = D #Depth of the layer
        self.F = F #Size of the filters "Spatial extent"
        self.S = S #Stride
        self.K = K #Number of filters
        self.P = P #Zero padding
        self.filters = Parameters(np.random.randn(self.K, self.D, self.F, self.F)) #Initialization of the filters
        self.b = Parameters(np.random.randn(self.K))


    def forward(self,inputs):
        self.H=inputs.shape[1]
        self.W=inputs.shape[2]
        patch = Patches(self.F, self.S)
        centers= patch.separate(self.zero_padding(inputs))
        H1 = int((self.H+2*self.P-self.F)/self.S +1)
        W1 = int((self.W+2*self.P-self.F)/self.S +1)
        D1 = self.K
        output = np.zeros([D1, H1, W1])
        for k in range(D1):

            count = 0
            for i in range(H1):
                
                for j in range(W1):
                    output[k,i,j] = np.sum(self.filters.data[k]*centers[count])+self.b.data[k]
                    count+=1
        return output


    def zero_padding(self,input):
        if self.P == 0:
            return input
        padding = np.zeros([self.D,self.H+2*self.P, self.W+2*self.P])
        padding[:, self.P:self.P+self.H, self.P:self.P+self.W] = input
        return padding
    

    def backward(self):

        pass

    def parameters(self):
        return[self.filters,self.b]
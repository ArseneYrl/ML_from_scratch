import numpy as np

class Patches:
    #Used in the CNN architecture for the forward and backward of the conv and maxpool layers
    def __init__(self, filter_size, stride):
        self.stride = stride
        self.filter_size = filter_size
        
    def separate(self, Input):
        self.X=Input
        #Input will be of shape (W, H, D)
        centers = []
        for i in range(0,Input.shape[1]-self.filter_size+1,self.stride):
            for j in range(0,Input.shape[2]-self.filter_size+1, self.stride):

                C=Input[:, i:i+self.filter_size, j:j+self.filter_size]
                centers.append(C)
        return centers
    
    def reconstruct_grad(self, centers):
        #reconstruct the gradient of the maxpool layers
        output = np.zeros(self.X.shape)
        count = 0
        for i in range(0,output.shape[1]-self.filter_size + 1,self.stride):

            for j in range(0,output.shape[2]-self.filter_size + 1,self.stride):

                output[:, i:i+self.filter_size, j:j+self.filter_size]+=centers[count]
                count += 1
        return output
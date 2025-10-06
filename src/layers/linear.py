import numpy as np

class Linear:
    def __init__(self,input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        std = np.sqrt(2.0 / self.input_size)

        self.W = np.random.normal(0, std, size=(self.input_size, self.output_size)) #Initialization of the weights
        self.b = np.zeros((1, self.output_size)) #Initialization of the biases

        self.grad_W = None 
        self.grad_b = None 
        

    def forward(self,X):
        #Feedforward
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, error):
        #Backward propagation
        batch_size = error.shape[0]

        self.grad_W = self.X.T @ error / batch_size
        self.grad_b = np.sum(error, axis=0, keepdims=True) / batch_size

        return error @ self.W.T
    
    def parameters(self):
        return [self.W, self.b]



    

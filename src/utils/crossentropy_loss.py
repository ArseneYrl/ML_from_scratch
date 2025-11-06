import numpy as np

def crossentropy_loss(ynew, ytrue):
    #Compute the crossentropy loss and its gradient
    exp = np.exp(ynew - np.max(ynew, axis=1, keepdims=True))
    softmax = exp / np.sum(exp, axis = 1, keepdims=True)

    loss = -np.mean(np.sum(ytrue * np.log(softmax + 1e-8), axis=1))

    batch_size = ynew.shape[0]
    grad = (softmax - ytrue) / batch_size

    return loss, grad
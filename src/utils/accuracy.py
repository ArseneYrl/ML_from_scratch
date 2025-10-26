import numpy as np

def accuracy(y_true, y_pred):
    if y_true.ndim == 1:
        y_true_fin = y_true
    else:
        y_true_fin = np.argmax(y_true, axis = 1) #If y is not one hot encoded
    return np.mean(y_true_fin == np.argmax(y_pred, axis = 1))
import numpy as np

def accuracy(y_true, y_pred):
    if y_true.dim == 1:
        y_true_fin = y_true
    else:
        y_true_fin = np.argmax(y_true, axis = 1)
    return np.mean(y_true_fin == np.argmax(y_pred, axis = 1))
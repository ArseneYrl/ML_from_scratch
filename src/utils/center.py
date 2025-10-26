import numpy as np

def center(Input,F, S):
    #Input will be of shape (W, H, D)
    centers = []
    rad = F // 2
    for i in range(rad,Input.shape[1]-rad,S):

        for j in range(rad,Input.shape[2]-rad,S):

            C=Input[:, i-rad:i+rad+1, j-rad:j+rad+1]
            centers.append(C)
    return centers
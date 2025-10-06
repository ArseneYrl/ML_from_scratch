import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.layers.linear import Linear


input_size=784
output_size=264

layer1=Linear(input_size, output_size)

X=np.random.randn(1000,input_size)
z=layer1.forward(X)

dZ = np.ones_like(z)

dX=layer1.backward(dZ)

print(layer1.parameters())
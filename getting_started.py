import numpy as np 
import numba 
from numba import cuda 

print(np.__version__)
print(numba.__version__)

cuda.detect() 

import numpy as np
from fealpy.mesh.core import lagrange_shape_function, lagrange_grad_shape_function

p = 2
n = 2

bc = np.array([1/3, 1/3, 1/3])
R0 = lagrange_shape_function(bc, p, n=n)
print('R0:', R0)
#R1 = lagrange_grad_shape_function(bc, p)
#print('R1:', R1)

import numpy as np
import sys

from fealpy.mesh import TriangleMesh
from fealpy.quadrature.TriangleQuadrature1 import TriangleQuadrature

k = int(sys.argv[1])

def u(p):
    x = p[..., 0]
    y = p[..., 1]
    val = x**k
    return val

def uu(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.exp(x+y)
    return val

def uuu(p):
    x = p[..., 0]
    y = p[..., 1]
    pi = np.float128('3.141592653589793238462643383279502884197')
    val = np.sin(pi*x)*np.sin(pi*y)
    return val

one = np.float128('1.0')
zero = np.float128('0.0')

node = np.array([
    (zero, zero),
    (one, zero),
    (one, one),
    (zero, one)], dtype=np.float128)
cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.uint32)
mesh = TriangleMesh(node, cell)
area = mesh.entity_measure('cell')


qf = TriangleQuadrature(k, ftype=node.dtype.type)
bcs, ws = qf.quadpts, qf.weights

p = mesh.bc_to_point(bcs)
val = uu(p)

a = np.float128('2.952492442012559756509852517869682817665821383151928174153192567074420535371962574241548128028788940')
b = np.einsum('i, ij, j->', ws, val, area)
print(np.abs(b - a))

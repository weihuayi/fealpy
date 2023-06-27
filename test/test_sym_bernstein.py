
import numpy as np
import sympy as sp

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import BernsteinFESpace
from fealpy.symcom import SimplexElementBasis 

p = 5

mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
space =  BernsteinFESpace(mesh, p=p)
space_sym = SimplexElementBasis(2, 'b')

bc = np.array([[1/3, 1/6, 1/2]])
phi = space.basis(bc)
phi_sym = space_sym.basis(p)
phi_sym_list = []
for _phi in phi_sym:
    ff = space_sym.sp_to_np_function(_phi)
    phi_sym_list.append(ff(1/3, 1/6, 1/2))
print(phi)
print(phi_sym_list)




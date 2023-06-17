import numpy as np
from fealpy.mesh import MeshFactory as MF
from fealpy.quadrature import GaussLegendreQuadrature, TensorProductQuadrature

mesh = MF.boxmesh2d([0, 1, 0, 1], nx=2, ny=2, meshtype='quad', p=2)

qf = GaussLegendreQuadrature(3)
qf = TensorProductQuadrature(qf, TD=2) 


a = np.array([[0.5, 0.5]], dtype=np.float)
b = np.array([[0.5, 0.5]], dtype=np.float)
bc = (a, b)

J = mesh.jacobi_matrix(bc)
print(J)



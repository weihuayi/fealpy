import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData,ToruSurfacesData,ElipsoidSurfaceData,HeartSurfacetData 
from fealpy.fem.SurfacePoissonFEMModel import SurfacePoissonFEMModel
from fealpy.quadrature import TriangleQuadrature 
from fealpy.tools.show import showmultirate

m = int(sys.argv[1])
p = int(sys.argv[2]) 
q = int(sys.argv[3])
if m == 1:
    pde = SphereSinSinSinData()
    mesh = pde.init_mesh()
elif m == 2:
    model = ToruSurfacesData() 
    surface = model.surface 
    mesh = surface.init_mesh()
elif m == 3:
    model = ElipsoidSurfaceData()
    surface = model.surface 
    mesh = surface.init_mesh()
elif m == 4:
    model = HeartSurfacetData()
    surface = model.surface 
    mesh = surface.init_mesh()

maxit = 4
errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{S,0}$',
             '$||\\nabla_S u - \\nabla_S u_h||_{S, 0}$'
             ]

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit):
    fem = SurfacePoissonFEMModel(mesh, pde, p, q)
    fem.solve()
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = fem.get_l2_error()
    errorMatrix[1, i] = fem.get_L2_error()
    errorMatrix[2, i] = fem.get_H1_semi_error()
    if i < maxit - 1:
        mesh.uniform_refine(1, pde.surface)

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix,  errorType)
plt.show()

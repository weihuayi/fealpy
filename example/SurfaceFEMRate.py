import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.model.surface_poisson_model_3d import SphereSinSinSinData,ToruSurfacesData,ElipsoidSurfaceData,HeartSurfacetData 
from fealpy.femmodel.SurfacePoissonFEMModel import SurfacePoissonFEMModel
from fealpy.quadrature import TriangleQuadrature 
from fealpy.tools.show import showmultirate


m = int(sys.argv[1])
p = int(sys.argv[2]) 
q = int(sys.argv[3])

if m == 1:
    model = SphereSinSinSinData()
    surface = model.surface 
    mesh = surface.init_mesh()
    mesh.uniform_refine(n=2, surface=surface)
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


integrator = TriangleQuadrature(q)
fem = SurfacePoissonFEMModel(mesh, surface, model, p=p, p0=p, integrator=integrator)
maxit = 4

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{S,0}$',
             '$||\\nabla_S u - \\nabla_S u_h||_{S, 0}$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit):
    fem.solve()
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    if i < maxit - 1:
        mesh.uniform_refine(1, surface)
        fem.reinit(mesh)

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix,  errorType)
plt.show()
plt.show()


import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.level_set_function import Sphere
from fealpy.model.surface_parabolic_model_3d import SinSinSinExpData
from fealpy.femmodel.SurfaceHeatFEMModel import SurfaceHeatFEMModel
from fealpy.quadrature import TriangleQuadrature 

from fealpy.tools.show import showmultirate


m = int(sys.argv[1])
p = int(sys.argv[2]) 
q = int(sys.argv[3])

if m == 1:
    model = SinSinSinExpData()
    surface = Sphere()
    mesh = surface.init_mesh()
    mesh.uniform_refine(n=0, surface=surface)


initTime = 0.0                                                    
stopTime = 1.0     
N = 4000

integrator = TriangleQuadrature(q)
fem = SurfaceHeatFEMModel(mesh, surface, model, initTime, stopTime, N, method='FM',integrator=integrator, p=p)
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros(maxit, dtype=np.float)
for i in range(maxit):
    fem.run()
    Ndof[i] = len(fem.uh)
    errorMatrix[i] = fem.maxError
    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)
print('error',errorMarrix)

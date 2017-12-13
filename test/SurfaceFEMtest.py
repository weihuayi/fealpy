import sys

import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.level_set_function import Sphere  
from fealpy.model.surface_poisson_model_3d import SphereSinSinSinData 
from fealpy.femmodel.SurfacePoissonFEMModel import SurfacePoissonFEMModel


def show_solution(mesh, uI):
    from mayavi import mlab
    point = mesh.point
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.points3d(point[:, 0], point[:, 1], point[:, 2], uI)
    mlab.show()


m = int(sys.argv[1])
p = int(sys.argv[2]) 

if m == 1:
    model = SphereSinSinSinData()

surface = Sphere()
mesh = surface.init_mesh()
mesh.uniform_refine(n=3, surface=surface)
fem = SurfacePoissonFEMModel(mesh, surface, model, p)
maxit = 4
error = np.zeros(maxit)
for i in range(maxit):
    fem.solve()
    error[i] = fem.l2_error()
    print(error[i])
    if i < maxit - 1:
        mesh.uniform_refine(1, surface)
        fem.reinit(mesh)
print(error[:-1]/error[1:])
#show_solution(fem.V.mesh, fem.uI)

#f = pl.figure()
#axes = a3.Axes3D(f)
#bc = mesh.barycenter()
#bc,_= surface.project(bc)
#color = model.solution(bc)
#mesh.add_plot(axes, cellcolor=color, showcolorbar=True)
#pl.show()

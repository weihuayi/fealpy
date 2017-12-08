import sys

import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.level_set_function import Sphere  
from fealpy.mesh.surface_mesh_generator import iso_surface 
from fealpy.model.surface_poisson_model_3d import SphereData 

from fealpy.femmodel.PoissonSurfaceFEMModel import SurfaceFEMModel
from fealpy.solver import solve

from fealpy.quadrature  import TriangleQuadrature

m = int(sys.argv[1])

if m == 1:
    model = SphereData()

qt = int(sys.argv[2]) 
n = int(sys.argv[3])
degree = int(sys.argv[4]) 

surface = Sphere()
mesh = iso_surface(surface, surface.box, nx=n, ny=n, nz=n)

V = function_space(mesh, 'Lagrange', degree)
uh = FiniteElementFunction(V)
Ndof = V.number_of_global_dofs()
a = SurfaceFEMModel(V,model,qfindex=1)
L = SurfaceFEMModel(V,model,qfindex=1)

fem = SurfaceFEMModel(V,model)

solve(fem,uh)








f = pl.figure()
axes = a3.Axes3D(f)
mesh.add_plot(axes)
pl.show()

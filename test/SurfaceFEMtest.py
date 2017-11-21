import sys

import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.surface_mesh_generator import iso_surface 

from fealpy.functionspace.tools import function_space
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.femmodel.PoissonSurfaceFEMModel import SurfaceFEMModel

qt = int(sys.argv[1]) 
n = int(sys.argv[2])
degree = int(sys.argv[3]) 

surface = Sphere()
mesh = iso_surface(surface, surface.box, nx=n, ny=n, nz=n)

V = function_space(mesh, 'Lagrange', degree)
uh = FiniteElementFunction(V)
Ndof = V.number_of_global_dofs()
a = SurfaceFEMModel(V,qfindex=1)



f = pl.figure()
axes = a3.Axes3D(f)
mesh.add_plot(axes)
pl.show()

import sys

import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.level_set_function import Sphere
from fealpy.functionspace.surface_lagrange_fem_space import SurfaceTriangleMesh,SurfaceLagrangeFiniteElementSpace
#from fealpy.functionspace.tools import function_space
#from fealpy.functionspace.function import FiniteElementFunction

from fealpy.femmodel.PoissonSurfaceFEMModel import SurfaceFEMModel

surface = Sphere()
smesh = surface.init_mesh()
smesh.uniform_refine(5, surface)

# surface mesh and surfaceTriangleMesh
stmesh = SurfaceTriangleMesh(surface, smesh, p=1)

## 怎么根据有限元函数找到 自由度组装矩阵，目前下面几行是错的
V = function_space(stmesh,'Lagrange',1)
uh = FiniteElementFunction(V)
a = get_stiff_matrix()
print(a) 





f = pl.figure()
axes = a3.Axes3D(f)
smesh.add_plot(axes)
pl.show()

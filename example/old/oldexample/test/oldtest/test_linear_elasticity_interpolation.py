
import numpy as np
import sys

from fealpy.functionspace.mixed_fem_space import HuZhangFiniteElementSpace
from fealpy.pde.linear_elasticity_model import PolyModel3d
from fealpy.fem.LinearElasticityFEMModel import LinearElasticityFEMModel 

import numpy as np  

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

pde = PolyModel3d()
mesh = pde.init_mesh(0)
space = HuZhangFiniteElementSpace(mesh, 4)
ipoints = space.interpolation_points()

print(pde.stress(ipoints))

axes = a3.Axes3D(pl.figure())
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints)
pl.show()

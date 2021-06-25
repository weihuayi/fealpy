#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh.SurfaceTriangleMeshOptAlg import SurfaceTriangleMeshOptAlg 
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh

from fealpy.decorator import cartesian, barycentric
from fealpy.pde.surface_poisson import SphereSimpleData  as PDE
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.tools.show import showmultirate, show_error_table

# solver
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat




p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(meshtype='tri', p=p) # p 次的拉格朗日四边形网格

for i in range(maxit):
    mesh.uniform_refine(n=n)
    space = ParametricLagrangeFiniteElementSpace(mesh, p=p)
    uI = space.interpolation(pde.solution)
    value = space.integral(uI)
    print(value)


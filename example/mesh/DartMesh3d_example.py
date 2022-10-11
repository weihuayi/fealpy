#!/usr/bin/env python3
# 
import numpy as np

from fealpy.geometry import CuboidDomain
from fealpy.geometry import huniform

from fealpy.mesh import DartMesh3d, TetrahedronMesh, MeshFactory
from fealpy.mesh import DistMesher3d 

def fh(p, *args):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    d = np.sqrt(x**2 + y**2 + z**2)
    h = hmin + d*0.1
    h[h>hmax] = hmax 
    return h

maxit = 1000
hmin = 0.1
domain = CuboidDomain()

mesher = DistMesher3d(domain, hmin, output=True)
mesh = mesher.meshing(maxit)
mesh = DartMesh3d.from_mesh(mesh)
mesh.celldata['cidx'] = np.arange(mesh.number_of_cells())
mesh.to_vtk(fname='cube.vtu')

mesh = mesh.dual_mesh(dual_point='circumcenter')
mesh.celldata['cidx'] = np.arange(mesh.number_of_cells())
mesh.to_vtk(fname='dual.vtu')


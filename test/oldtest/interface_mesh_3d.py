import numpy as np

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.level_set_function import HeartSurface
from fealpy.mesh.level_set_function import OrthocircleSurface
from fealpy.mesh.level_set_function import QuarticsSurface 

from fealpy.mesh.vtkMeshIO import write_vtk_mesh 

from fealpy.mesh.interface_mesh_generator import InterfaceMesh3d 

import time


def my_test(n, interface, fname):

    t0  = time.clock()
    intmesh = InterfaceMesh3d(interface, interface.box, n)
    pmesh = intmesh.run()
    t1 = time.clock()

    pmesh.check()
    a = pmesh.face_angle()
    write_vtk_mesh(pmesh, fname)

    return t1 - t0, np.max(a), (n+1)*(n+1)*(n+1)

#interface = Sphere()
interface = HeartSurface()
#interface = OrthocircleSurface()
interface = QuarticsSurface()
N = np.array([20, 40, 80, 160])
maxit = len(N)
T = np.zeros(maxit, dtype=np.float)
A = np.zeros(maxit, dtype=np.float)
Ndof = np.zeros(maxit, dtype=np.int)
for i, n in enumerate(N):
    T[i], A[i], Ndof[i] = my_test(n, interface, 'test.vtk')

print(T[1:]/T[0:-1])
print('Time:', T)
print('Angle:', A)
print('Ndof:', Ndof)



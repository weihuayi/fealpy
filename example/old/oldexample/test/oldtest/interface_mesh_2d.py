
import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.interface_mesh_generator import InterfaceMesh2d
from fealpy.mesh.PolygonMesh import PolygonMesh 
from fealpy.mesh.vtkMeshIO import write_vtk_mesh 

from fealpy.mesh.curve import CircleCurve
from fealpy.mesh.curve import Curve1
from fealpy.mesh.curve import Curve2 
from fealpy.mesh.curve import Curve3

import time

def show_mesh(mesh):
    fig = plt.figure()
    axes = fig.gca() 
    mesh.add_plot(axes, cellcolor=[0.5, 0.9, 0.45])
    plt.show()

def my_test(n, interface, fname):

    t0  = time.clock()
    alg = InterfaceMesh2d(interface, interface.box, n)
    ppoint, pcell, pcellLocation = alg.run()
    t1 = time.clock()

    pmesh = PolygonMesh(ppoint, pcell, pcellLocation)
    a = pmesh.angle()
    write_vtk_mesh(pmesh, fname)

    return t1 - t0, np.max(a), (n+1)*(n+1)

N = [100, 200, 400, 800, 1600]

interface = Curve1(a=6)
interface = Curve3()
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


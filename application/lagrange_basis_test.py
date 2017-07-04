import sys
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.functionspace.tools import function_space

inputName = sys.argv[1]
degree = int(sys.argv[2])
outputName = sys.argv[3]

data = sio.loadmat(inputName)

point = data['point']
cell = data['cell'] - 1

mesh = TriangleMesh(point, cell)
edge = mesh.ds.edge
cell2edge = mesh.ds.cell_to_edge()
cell2edgeSign = mesh.ds.cell_to_edge_sign()
V = function_space(mesh, 'Lagrange', degree)

ipoint = V.interpolation_points()
edge2dof = V.edge_to_dof()
cell2dof = V.cell_to_dof()

data = {
        'point':point, 
        'cell':cell + 1, 
        'edge':edge + 1,
        'cell2edge':cell2edge + 1,
        'cell2edgeSign':cell2edgeSign,
        'dofPoint':ipoint, 
        'edge2dof':edge2dof + 1,
        'cell2dof':cell2dof + 1
        }
sio.matlab.savemat(outputName, data)

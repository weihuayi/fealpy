import numpy as np

from fealpy.mesh import LagrangeHexahedronMesh, TetrahedronMesh
from cssim import GRDECLReader

reader = GRDECLReader('../repository/cssim/data/10.GRDECL')

node = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
    [1, 1, 1], [0, 1, 0], [0, 1, 1], [2, 0, 0], [2, 0, 1], [2, 1, 0], 
    [2, 1, 1]], dtype=np.float_)
cell = np.array([[0, 1, 2, 3, 6, 7, 4, 5], [2, 3, 8, 9, 4, 5, 10, 11]], dtype=np.int_)
flag = node[:, 2] >0.9999
node[flag, 2]*=20

mesh = LagrangeHexahedronMesh(node, cell, p=1)
edge = mesh.entity('edge')
face = mesh.entity('face')
mesh.ds.NF=len(face)

mesh = reader.to_tetmesh(mesh)

node = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 20],
    [1, 0, 20],
    [1, 1, 20],
    [0, 1, 20]], dtype=np.float)
"""
node = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]], dtype=np.float)

"""
cell = np.array([
    [0, 1, 2, 6],
    [0, 5, 1, 6],
    [0, 4, 5, 6],
    [0, 7, 4, 6],
    [0, 3, 7, 6],
    [0, 2, 3, 6]], dtype=np.int)
mesh = TetrahedronMesh(node, cell)
mesh.uniform_bisect(2)
mesh.to_vtk(fname='112ew.vtu')



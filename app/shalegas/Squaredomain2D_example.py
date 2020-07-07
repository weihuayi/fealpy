#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian

from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace.femdof import multi_index_matrix2d


class CrackCosCosData:
    """
    -\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=0, meshtype='tri'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(2)

        NN = mesh.number_of_nodes()
        node = np.zeros((NN+3, 2), dtype=np.float64)
        node[:NN] = mesh.entity('node')
        node[NN:] = node[[5], :]
        cell = mesh.entity('cell')

        cell[13][cell[13] == 5] = NN
        cell[18][cell[18] == 5] = NN

        cell[19][cell[19] == 5] = NN+1
        cell[12][cell[12] == 5] = NN+1

        cell[6][cell[6] == 5] = NN+2
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh


    @cartesian
    def solution(self, p):
        return 0.0


    @cartesian
    def source(self, p):
        val = np.array([0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape)

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0
        val[..., 1] = 0
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1])
        flag0 = np.abs(x) < 1e-13
        val[flag0] = -1
        flag1 = np.abs(x-1) < 1e-13
        val[flag1] = 1
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x) < 1e-13
        return flag

    @cartesian
    def is_neumann_bc(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x - 0) < 1e-13) | (np.abs(x - 1) < 1e-13)
        return flag


n = int(sys.argv[1])
p = int(sys.argv[2])

pde = CrackCosCosData()

mesh = pde.init_mesh(n=n, meshtype='tri')
space = RaviartThomasFiniteElementSpace2d(mesh, p=p)

udof = space.number_of_global_dofs()
pdof = space.smspace.number_of_global_dofs()
gdof = udof + pdof

uh = space.function()
ph = space.smspace.function()
A = space.stiff_matrix()
B = space.div_matrix()
F1 = space.source_vector(pde.source)
AA = bmat([[A, -B], [-B.T, None]], format='csr')
isBdDof = space.set_dirichlet_bc(uh, pde.neumann, threshold=pde.is_neumann_bc)
x = np.r_['0', uh, ph] 
isBdDof = np.r_['0', isBdDof, np.zeros(pdof, dtype=np.bool_)]

FF = np.r_['0', np.zeros(udof, dtype=np.float64), F1]

FF -= AA@x
bdIdx = np.zeros(gdof, dtype=np.int)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof)
T = spdiags(1-bdIdx, 0, gdof, gdof)
AA = T@AA@T + Tbd
FF[isBdDof] = x[isBdDof]
x[:] = spsolve(AA, FF)
uh[:] = x[:udof]
ph[:] = x[udof:]

box = [0, 1, 0, 1]
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, box=box)
#mesh.find_node(axes, showindex=True)
#mesh.find_edge(axes, showindex=True)
#mesh.find_cell(axes, showindex=True)
node = mesh.entity_barycenter('cell')
qf = mesh.integrator(1)
bcs, ws = qf.get_quadrature_points_and_weights()
ps = mesh.bc_to_point(bcs)
phi = space.basis(bcs)
phi1 = np.einsum('i, ijkm->jkm', ws, phi, optimize=True)
edge2cell = mesh.ds.edge_to_cell()
cell2edge = mesh.ds.cell_to_edge()
u1 = uh[cell2edge[:, 0]].reshape((-1,1))
u2 = uh[cell2edge[:, 1]].reshape((-1,1))
u2 = np.hstack((u1, u2))
u3 = uh[cell2edge[:, 2]].reshape((-1,1))
u3 = np.hstack((u2, u3))
uv = np.zeros(node.shape)
ws = np.array([1/2, np.sqrt(2)/2, np.sqrt(2)/2])
uv[..., 0] = np.einsum('j, ij, ij->i', ws, u3, phi1[..., 0], optimize=True)
uv[..., 1] = np.einsum('j, ij, ij->i', ws, u3, phi1[..., 1], optimize=True)
uv = uv.reshape(-1, 2)
print(node.shape)
print(uv.shape)
axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1])
plt.show()



 

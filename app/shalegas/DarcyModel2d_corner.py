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


class LeftRightData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh


    @cartesian
    def pressure(self, p):
        pass

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def gradient(self, p):
        pass

    @cartesian
    def flux(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = (np.abs(x) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x < 1/16)
        val[flag0 | flag1] = 10

        flag0 = (np.abs(x-1) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x > 1 - 1/16)
        val[flag1] = -10
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        pass


n = int(sys.argv[1])
p = int(sys.argv[2])

pde = LeftRightData()
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

isBdDof = space.set_dirichlet_bc(uh, pde.neumann)
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

area = mesh.entity_measure('cell')
x[udof:] -= sum(x[udof:]*area)/sum(area)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor=x[udof:], showcolorbar=True)
node = mesh.entity('node')
bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
ps = mesh.bc_to_point(bc)
V = uh.value(bc)
axes.quiver(ps[:, 0], ps[:, 1], V[:, 0], V[:, 1], angles='xy', units='xy')
axes.set_axis_on()
plt.show()

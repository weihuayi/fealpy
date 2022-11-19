import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.mesh import MeshFactory as mf

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import cartesian

class moudle:
    def __init__(self):
        pass

    def domain(self):
        val = [-1, 1, -1, 1]
        return val

    @cartesian
    def materal(self, p):
        shape = p.shape + (2,)
        val = np.zeros(shape=shape)
        val[..., 0, 0] = val[..., 1, 1] = 1
        return val

    @cartesian
    def source(self, p):
        val = 0
        return val


    @cartesian
    def dirichlet(self, p):
        x = p[..., 0]
        val = np.zeros_like(x)
        flag = np.abs(x+1) < 1e-12
        val[flag] = 900
        flag = np.abs(x - 1) < 1e-12
        val[flag] = 293.15
        return val

    def is_dirichlet(self, p):
        x = p[..., 0]
        flag = (np.abs(x + 1) < 1e-12) | (np.abs(x - 1) < 1e-12)
        return flag 



pde = moudle()
# mesh = mf.triangle(box=pde.domain(), h=0.008)
mesh = mf.boxmesh2d(pde.domain(), nx=40, ny=40)
space = LagrangeFiniteElementSpace(mesh, p=1)

if False:
    node = mesh.entity('node')
    isLeftNode = pde.is_left_Dirichlet(node)

    fig = plt.figure() 
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, index=isLeftNode)
    plt.show()


A = space.stiff_matrix(c=pde.materal)
F = space.source_vector(f=pde.source)
uh = space.function()

# 左边界
bc = DirichletBC(space,
                  pde.dirichlet,
                  threshold=pde.is_dirichlet)

A, F = bc.apply(A, F, uh)

uh[:] = spsolve(A, F)
bc = np.array([1 / 3, 1 / 3, 1 / 3])
val = uh(bc)
mesh.add_plot(plt, cellcolor=val, linewidths=0, showcolorbar=True)
uh.add_plot(plt, cmap='jet')
plt.show()

#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator, ScalarConvectionIntegrator, DirichletBC
from fealpy.fem import BilinearForm, LinearForm
from fealpy.pde.diffusion_convection_reaction import PMLPDEModel2d


#定义波数、入射波方向、入射波、求解域
k = 10
d = [0, -1]
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'
domain = [-6, 6, -6, 6]
#用于判断在散射体内部或外部的水平集函数
def levelset(p):
    ctr = np.array([[0, -1.0]])
    return np.linalg.norm(p - ctr, axis=-1) - 0.3

mesh = TriangleMesh.interfacemesh_generator(box=domain, nx=150, ny=150, phi=levelset)
p=1
qf = mesh.integrator(p+3, 'cell')
bc_, ws = qf.get_quadrature_points_and_weights()
qs = len(ws)

#实例化带PML的pde
pml = PMLPDEModel2d(levelset=levelset,
                 domain=domain,
                 u_inc=u_inc,
                 A=1,
                 k=k, 
                 d=d, 
                 refractive_index=[1, 1+1/k**2],
                 absortion_constant=1.79,
                 lx=1.0,
                 ly=1.0)

#有限元求解前向散射问题
space = LagrangeFESpace(mesh, p=p)
space.ftype = complex

D = ScalarDiffusionIntegrator(c=pml.diffusion_coefficient, q=p+3)
C = ScalarConvectionIntegrator(c=pml.convection_coefficient, q=p+3)
M = ScalarMassIntegrator(c=pml.reaction_coefficient, q=p+3)
f = ScalarSourceIntegrator(pml.source, q=p+3)

b = BilinearForm(space)
b.add_domain_integrator([D, C, M])

l = LinearForm(space)
l.add_domain_integrator(f)

A = b.assembly()
F = l.assembly()

bc = DirichletBC(space, pml.dirichlet) 
uh = space.function(dtype=np.complex128)
A, F = bc.apply(A, F, uh)
uh[:] = spsolve(A, F)

#可视化散射场数据
fig = plt.figure()

value = uh(bc_)
mesh.ftype = np.float64
mesh.add_plot(plt, cellcolor=value[0, ...].real, linewidths=0)
mesh.add_plot(plt, cellcolor=value[0, ...].imag, linewidths=0)

axes = fig.add_subplot(1, 3, 1)
mesh.add_plot(axes)
axes = fig.add_subplot(1, 3, 2, projection='3d')
mesh.show_function(axes, np.real(uh))
axes = fig.add_subplot(1, 3, 3, projection='3d')
mesh.show_function(axes, np.imag(uh))

plt.show()

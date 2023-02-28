#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.linear_elasticity_model import  BoxDomainData3d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.boundarycondition import NeumannBC

from scipy.sparse.linalg import spsolve

from timeit import default_timer as timer


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--GD',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=2, type=int,
        help='初始网格加密的次数, 默认初始加密 2 次.')

parser.add_argument('--scale',
        default=1, type=float,
        help='网格变形系数，默认为 1')


args = parser.parse_args()
p = args.degree
GD = args.GD
n = args.nrefine
scale = args.scale


if GD == 2:
    from fealpy.pde.linear_elasticity_model import BoxDomainData2d as PDE
elif GD == 3:
    from fealpy.pde.linear_elasticity_model import BoxDomainData3d as PDE

pde = PDE()
mesh = pde.init_mesh(n=n)
NN = mesh.number_of_nodes()

space = LagrangeFiniteElementSpace(mesh, p=p)



uh = space.function(dim=GD) # (NDof, GD)
A = space.linear_elasticity_matrix(pde.lam, pde.mu, q=p+2)
F = space.source_vector(pde.source, dim=GD)

if hasattr(pde, 'neumann'):
    print('neumann')
    bc = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)
    F = bc.apply(F)

if hasattr(pde, 'dirichlet'):
    print('dirichlet')
    bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    A, F = bc.apply(A, F, uh)

uh.T.flat[:] = spsolve(A, F)

# 画出原始网格
mesh.add_plot(plt)

# 画出变形网格
mesh.node += scale*uh[:NN]
mesh.add_plot(plt)

plt.show()

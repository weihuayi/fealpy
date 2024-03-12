import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData
from fealpy import logger
from fealpy.jax.mesh import TriangleMesh 
from fealpy.jax.functionspace import LagrangeFESpace
from fealpy.fem.diffusion_integrator import DiffusionIntegrator 


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

pde = CosCosData()
domain = pde.domain()

errorType = ['$|| u - u_h||_{\\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

mesh = TriangleMesh.from_box(box = domain, nx = nx*2**i, ny = ny*2**i)
space = LagrangeFESpace(mesh, p = p)


print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

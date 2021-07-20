#!/usr/bin/env python3
#

import argparse

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve

from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上基于 Log Marking 策略的自适应方法
        """)

parser.add_argument('--theta',
        default=0.2, type=float,
        help='自适应 theta 控制参数.')

parser.add_argument('--atype',
        default='log', type=str,
        help='自适应类型，默认为 log 策略.')

parser.add_argument('--maxdof',
        default=200000, type=int,
        help='默认网格自适应加密最大自由度个数, 默认最大自由度个数为 200000')

args = parser.parse_args()

theta = args.theta
maxdof = args.maxdof
atype = args.atype

pde = LShapeRSinData()

tritree = pde.init_mesh(n=4, meshtype='tritree')

errorType = ['$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$']

NDof = [] 
errorMatrix = [[], []] 

k = 0
while True:
    mesh = tritree.to_conformmesh()

    fname = './test-' + str(k) + '.png'
    mesh.add_plot(plt)
    plt.savefig(fname)
    plt.close()

    space = LagrangeFiniteElementSpace(mesh, p=1)
    A = space.stiff_matrix(q=2)
    F = space.source_vector(pde.source)

    NDof += [space.number_of_global_dofs()]
    bc = DirichletBC(space, pde.dirichlet) 

    uh = space.function()
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)
    rguh = space.grad_recovery(uh)

    errorL1 = space.integralalg.error(pde.solution, uh.value)
    errorH1 = space.integralalg.error(pde.gradient, uh.grad_value)

    errorMatrix[0] += [errorL1]
    errorMatrix[1] += [errorH1]

    if NDof[-1] < maxdof:
        eta = space.recovery_estimate(uh)

        if atype == 'log':
            options = tritree.adaptive_options()
            tritree.adaptive(eta, options)
        elif atype == 'L2':
            isMarkedCell = tritree.refine_marker(eta, theta, 'L2')
            tritree.refine_1(isMarkedCell=isMarkedCell)
        k += 1
    else:
        break

NDof = np.array(NDof)
errorMatrix = np.array(errorMatrix)


showmultirate(plt, k - 5, NDof, errorMatrix, errorType)
plt.show()

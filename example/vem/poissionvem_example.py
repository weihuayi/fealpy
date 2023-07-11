import argparse 
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import ipdb

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import TriangleMesh 
from fealpy.mesh import PolygonMesh

from fealpy.functionspace import ConformingScalarVESpace2d

from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ConformingScalarVEMH1Projector2d
from fealpy.vem import ConformingScalarVEML2Projector2d 
from fealpy.vem import ConformingScalarVEMLaplaceIntegrator2d
from fealpy.vem import ConformingVEMScalarSourceIntegrator2d
from fealpy.vem import BilinearForm
from fealpy.vem import LinearForm

from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        多边形网格上的任意次协调虚单元方法  
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='虚单元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='x 方向剖分段数， 默认 4 段.')

parser.add_argument('--ny',
        default=4, type=int,
        help='y 方向剖分段数， 默认 4 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

pde = CosCosData()
domain = pde.domain()


errorType = ['$|| u - \Pi u_h||_{\Omega,0}$',
             '$||\\nabla u - \Pi \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

for i in range(maxit):
    tmesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)

    space = ConformingScalarVESpace2d(mesh, p=degree)
    uh = space.function()
    
    NDof[i] = space.number_of_global_dofs()
  
    #组装刚度矩阵 A 
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)

    d = ConformingVEMDoFIntegrator2d()
    D = d.assembly_cell_matrix(space, M)

    projector = ConformingScalarVEMH1Projector2d(D)
    PI1 = projector.assembly_cell_matrix(space)

    a = BilinearForm(space)
    I = ConformingScalarVEMLaplaceIntegrator2d(projector)
    a.add_domain_integrator(I)
    A = a.assembly()

    #组装右端 F
    a = ConformingScalarVEML2Projector2d(M, PI1)
    PI0 = a.assembly_cell_matrix(space)

    b = ConformingVEMScalarSourceIntegrator2d(pde.source, PI0)
    a = LinearForm(space)
    a.add_domain_integrator(b)
    F = a.assembly()

    #处理边界 
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F).reshape(-1)
    sh = space.project_to_smspace(uh, PI1)

    errorMatrix[0, i] = mesh.error(pde.solution, sh.value)
    errorMatrix[1, i] = mesh.error(pde.gradient, sh.grad_value)

    #uI = space.interpolation(pde.solution)
    nx *= 2
    ny *= 2
    
fig = plt.figure()
axes = fig.gca()
linetype = ['k-*', 'r-o', 'b-D', 'g-->', 'k--8', 'm--x','r-.x']
c = np.polyfit(np.log(NDof), np.log(errorMatrix[0]), 1)
axes.loglog(NDof, errorMatrix[0], linetype[0], label='$||u-\Pi u_h||_\\infty = O(h^{%0.4f})$'%(c[0]))

c = np.polyfit(np.log(NDof), np.log(errorMatrix[1]), 1)
axes.loglog(NDof, errorMatrix[1], linetype[1], label='$|| \\nabla u - \Pi \\nabla u_h||_0 = O(h^{%0.4f})$'%(c[0]))



axes.legend()
"""
mesh.add_plot(plt)
uh.add_plot(plt, cmap='rainbow')
"""
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20, lw=2, ms=4)

plt.show()

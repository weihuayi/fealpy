import argparse 
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# 模型数据
from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.poisson_2d import LShapeRSinData

# 网格 
from fealpy.mesh import PolygonMesh
from fealpy.mesh import HalfEdgeMesh2d

# 非协调空间
from fealpy.functionspace import NonConformingScalarVESpace2d

# 积分子
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import NonConformingVEMDoFIntegrator2d
from fealpy.vem import NonConformingScalarVEMH1Projector2d
from fealpy.vem import NonConformingScalarVEML2Projector2d 
from fealpy.vem import NonConformingScalarVEMLaplaceIntegrator2d
from fealpy.vem import NonConformingVEMScalarSourceIntegrator2d
from fealpy.vem import PoissonCVEMEstimator

# 双线性型
from fealpy.vem import BilinearForm

# 线性型
from fealpy.vem import LinearForm

from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        多边形网格上的任意次非协调虚单元方法  
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
        default=8, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

#pde = CosCosData()
pde = LShapeRSinData()
domain = pde.domain()

errorType = ['$|| u - \Pi u_h||_{\Omega,0}$',
             '$||\\nabla u - \Pi \\nabla u_h||_{\Omega, 0}$',
             '$\eta$']
errorMatrix = np.zeros((3, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

#mesh = PolygonMesh.from_box(domain, nx=nx, ny=ny)
mesh = pde.init_mesh(n = 1, meshtype='quad')
mesh = PolygonMesh.from_mesh(mesh)
Hmesh = HalfEdgeMesh2d.from_mesh(mesh)
fig = plt.figure()
axes  = fig.gca()
mesh.add_plot(axes)
plt.show()

for i in range(maxit):
    #mesh = pde.init_mesh(n = i+1, meshtype='tri')
    #mesh = PolygonMesh.from_mesh(mesh)
    space = NonConformingScalarVESpace2d(mesh, p=degree)
    uh = space.function()
    
    NDof[i] = space.number_of_global_dofs()
    print(NDof[i]) 
    #组装刚度矩阵 A 
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)

    d = NonConformingVEMDoFIntegrator2d()
    D = d.assembly_cell_matrix(space, M)

    h1 = NonConformingScalarVEMH1Projector2d(D)
    PI1 = h1.assembly_cell_matrix(space)
    G = h1.G

    li = NonConformingScalarVEMLaplaceIntegrator2d(PI1, G, D)
    bform = BilinearForm(space)
    bform.add_domain_integrator(li)
    A = bform.assembly()

    #组装右端 F
    l2 = NonConformingScalarVEML2Projector2d(M, PI1)
    PI0 = l2.assembly_cell_matrix(space)

    si = NonConformingVEMScalarSourceIntegrator2d(pde.source, PI0)
    lform = LinearForm(space)
    lform.add_domain_integrator(si)
    F = lform.assembly()

    #处理边界 
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)
    sh = space.project_to_smspace(uh, PI1)

    estimator = PoissonCVEMEstimator(space, M, PI1)
    eta = estimator.residual_estimate(uh, pde.source)
    
    errorMatrix[0, i] = mesh.error(pde.solution, sh.value)
    errorMatrix[1, i] = mesh.error(pde.gradient, sh.grad_value)
    errorMatrix[2, i] = np.sqrt(np.sum(eta))
    options = Hmesh.adaptive_options(HB=None)
    Hmesh.adaptive(eta, options)
    newcell, cellocation = Hmesh.entity('cell')
    newnode = Hmesh.entity("node")[:]
    mesh = PolygonMesh(newnode, newcell, cellocation)
showmultirate(plt, maxit-2, NDof, errorMatrix, errorType, propsize=20, lw=2, ms=4)
plt.show()
'''
fig1 = plt.figure()
axes  = fig1.gca()
mesh.add_plot(axes)
plt.show()
'''



  

import argparse 
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import copy

# 模型数据
from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.poisson_2d import LShapeRSinData

# 网格
from fealpy.mesh import PolygonMesh
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d

# 协调有限元空间
from fealpy.functionspace import ConformingScalarVESpace2d

# 积分子
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ConformingScalarVEMH1Projector2d
from fealpy.vem import ConformingScalarVEML2Projector2d 
from fealpy.vem import ConformingScalarVEMLaplaceIntegrator2d
from fealpy.vem import ConformingVEMScalarSourceIntegrator2d
from fealpy.vem import PoissonCVEMEstimator

# 双线性型
from fealpy.vem import BilinearForm

# 线性型
from fealpy.vem import LinearForm

# 边界条件
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

#pde = CosCosData()
pde = LShapeRSinData()
domain = pde.domain()


errorType = ['$|| u - \Pi u_h||_{\Omega,0}$',
             '$||\\nabla u - \Pi \\nabla u_h||_{\Omega, 0}$',
             '$\eta $']
errorMatrix = np.zeros((3, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

#mesh = PolygonMesh.from_box(domain, nx=nx, ny=ny)
mesh = pde.init_mesh(n = 1, meshtype='quad')
#mesh = PolygonMesh.from_mesh(mesh)
Hmesh = HalfEdgeMesh2d.from_mesh(mesh)
fig = plt.figure()
axes  = fig.gca()
Hmesh.add_plot(axes)
#plt.show()

for i in range(maxit):
    #mesh = PolygonMesh.from_box(domain, nx=nx, ny=ny)
    #mesh = pde.init_mesh(n = i+1, meshtype='tri')
    #mesh = PolygonMesh.from_mesh(mesh)
    space = ConformingScalarVESpace2d(Hmesh, p=degree)
    uh = space.function()
    
    NDof[i] = space.number_of_global_dofs()
  
    #组装刚度矩阵 A 
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)

    d = ConformingVEMDoFIntegrator2d()
    D = d.assembly_cell_matrix(space, M)

    h1 = ConformingScalarVEMH1Projector2d(D)
    PI1 = h1.assembly_cell_matrix(space)
    G = h1.G

    li = ConformingScalarVEMLaplaceIntegrator2d(PI1, G, D)
    bform = BilinearForm(space)
    bform.add_domain_integrator(li)
    A = bform.assembly()

    #组装右端 F
    l2 = ConformingScalarVEML2Projector2d(M, PI1)
    PI0 = l2.assembly_cell_matrix(space)

    si = ConformingVEMScalarSourceIntegrator2d(pde.source, PI0)
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
    
    errorMatrix[0, i] = Hmesh.error(pde.solution, sh.value)
    errorMatrix[1, i] = Hmesh.error(pde.gradient, sh.grad_value)
    errorMatrix[2, i] = np.sqrt(np.sum(eta))
    options = Hmesh.adaptive_options(HB=None)
    Hmesh.adaptive(eta, options)
    newcell = Hmesh.entity('cell')
    newnode = Hmesh.entity("node")[:]
    #mesh = PolygonMesh(newnode, newcell)
showmultirate(plt, maxit-4, NDof, errorMatrix, errorType, propsize=20, lw=2, ms=4)
print(errorMatrix)
plt.show()
fig1 = plt.figure()
axes  = fig1.gca()
Hmesh.add_plot(axes)
plt.show()

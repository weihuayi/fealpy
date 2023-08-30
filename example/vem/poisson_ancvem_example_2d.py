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
from fealpy.mesh.adaptive_tools import mark
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
        default=4, type=int,
        help='虚单元空间的次数, 默认为 1 次.')

parser.add_argument('--maxit',
        default=400, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--theta',
        default=0.2, type=int,
        help='自适应参数， 默认0.2')
args = parser.parse_args()

degree = args.degree
maxit = args.maxit
theta = args.theta


#pde = CosCosData()
pde = LShapeRSinData()
domain = pde.domain()

errorType = ['$|| u - \Pi u_h||_{0,\Omega}$',
             '$||\\nabla u -  \\nabla \Pi u_h||_{0, \Omega}$',
             '$\eta$']

errorMatrix = np.zeros((3, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

mesh = pde.init_mesh(n = 1, meshtype='quad')
mesh = PolygonMesh.from_mesh(mesh)
Hmesh = HalfEdgeMesh2d.from_mesh(mesh)

fig = plt.figure()
axes  = fig.gca()
mesh.add_plot(axes)
plt.show()

for i in range(maxit):
    space = NonConformingScalarVESpace2d(mesh, p=degree)
    uh = space.function()
    
    NDof[i] = space.number_of_global_dofs()
    
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
    
    print(i,":",NDof[i],",",np.sqrt(np.sum(eta))) 
    errorMatrix[0, i] = mesh.error(pde.solution, sh.value)
    errorMatrix[1, i] = mesh.error(pde.gradient, sh.grad_value)
    errorMatrix[2, i] = np.sqrt(np.sum(eta))
    
    isMarkedCell = mark(eta, theta, 'L2')
    Hmesh.adaptive_refine(isMarkedCell, method='poly')
    newcell, cellocation = Hmesh.entity('cell')
    newnode = Hmesh.entity("node")[:]
    mesh = PolygonMesh(newnode, newcell, cellocation)
    if NDof[i] > 1e4 :
        iterations = i 
        break
    ''' 
    if np.sqrt(np.sum(eta)) < 1e-3 :
        print("eta", np.sqrt(np.sum(eta)))
        iterations = i 
        break
    '''
    
    '''log 加密策略
    options = Hmesh.adaptive_options(HB=None)
    Hmesh.adaptive(eta, options)
    newcell, cellocation = Hmesh.entity('cell')
    newnode = Hmesh.entity("node")[:]
    mesh = PolygonMesh(newnode, newcell, cellocation)
    '''

#showmultirate(plt, maxit-10, NDof, errorMatrix, 
#        errorType, propsize=20, lw=2, ms=4)

showmultirate(plt, iterations-20, NDof[:iterations], errorMatrix[:,:iterations], 
        errorType, propsize=20, lw=2, ms=4)
np.savetxt("Ndof.txt", NDof[:iterations], delimiter=',')
np.savetxt("errorMatrix.txt", errorMatrix[:,:iterations], delimiter=',')

plt.xlabel('Number of d.o.f', fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel('Error', fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.savefig('error.jpg')
plt.show()

fig1 = plt.figure()
axes  = fig1.gca()
mesh.add_plot(axes)
plt.show()


  

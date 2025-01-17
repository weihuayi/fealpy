import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# 网格
from ..mesh import PolygonMesh
from ..mesh.halfedge_mesh import HalfEdgeMesh2d

# 协调有限元空间
from ..functionspace import ConformingScalarVESpace2d

# 积分子
from . import ScaledMonomialSpaceMassIntegrator2d
from . import ConformingVEMDoFIntegrator2d
from . import ConformingScalarVEMH1Projector2d
from . import ConformingScalarVEML2Projector2d
from . import ConformingScalarVEMLaplaceIntegrator2d
from . import ConformingVEMScalarSourceIntegrator2d
from . import PoissonCVEMEstimator

# 双线性型
from . import BilinearForm

# 线性型
from . import LinearForm

# 边界条件
from ..boundarycondition import DirichletBC

from ..mesh.adaptive_tools import mark


class PoissonACVEMSolver:
    def __init__(self, pde, mesh, p=1):
        '''
        初始化 PoissonACVEMSolver 对象

        Parameters:
            pde: 模型问题
            mesh: 初始网格
            p: 网格阶数
        '''
        self.mesh = PolygonMesh.from_mesh(mesh)
        self.Hmesh = HalfEdgeMesh2d.from_mesh(mesh)
        self.pde = pde
        self.p = p

        self.NDof = []
        self._NF = 0 # 函数solve调用计数器

    def solve(self,save_data=False,filepath='./'):
        '''
        组装矩阵与右端项，求解方程
        Parameters:
            save_data: 是否保存数据，默认为 False
            filepath: 数据保存路径，默认为当前目录下
        Returns:
            uh: 数值解
        '''
        mesh = self.mesh
        pde = self.pde
        space = ConformingScalarVESpace2d(mesh, p=self.p)
        uh = space.function()

        self.NDof.append(space.number_of_global_dofs())
      
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
        if save_data:
            np.savez(filepath+f"equ{self._NF}.npz",A=A,F=F,uh=uh)

        uh.M = M
        uh.PI1 = PI1
        self._NF += 1
        return uh

    def adaptive_solve(self, maxit=40, theta=0.2, method='L2',save_data=False,meshfilepath="./",equfilepath="./"):
        '''
        自适应求解

        Parameters:
            maxit: 最大迭代次数，默认为 40
            theta: 标记策略参数，默认为 1.0
            save_data: 是否保存数据，默认为 False
            meshfilepath: 网格数据保存路径，默认为当前目录下
            equfilepath: 求解方程数据保存路径，默认为当前目录下

        Returns:
            None
        '''
        errorMatrix = np.zeros((3,maxit),dtype=np.float64)
        Hmesh = self.Hmesh
        pde = self.pde
        for i in range(maxit):
            uh = self.solve(save_data=save_data,filepath=equfilepath)
            space = uh.space
            sh = space.project_to_smspace(uh, uh.PI1)

            estimator = PoissonCVEMEstimator(space, uh.M, uh.PI1)
            eta = estimator.residual_estimate(uh, pde.source)
            
            errorMatrix[0, i] = self.mesh.error(pde.solution, sh.value)
            errorMatrix[1, i] = self.mesh.error(pde.gradient, sh.grad_value)

            errorMatrix[2, i] = np.sqrt(np.sum(eta))
            # L2 标记策略加密
            if method=='L2':
                isMarkedCell = mark(eta, theta=theta)
                Hmesh.adaptive_refine(isMarkedCell, method='poly')
            # Log 标记策略
            elif method=='Log':
                options = Hmesh.adaptive_options(theta=theta,HB=None)
                Hmesh.adaptive(eta, options)
            else:
                raise ValueError("Wrong marking method!\"L2\" or \"Log\"")

            newcell = Hmesh.entity('cell')
            newnode = Hmesh.entity("node")[:]
            self.mesh = PolygonMesh(newnode, newcell)
            # 保存网格数据
            if save_data:
                cell = np.concatenate(newcell)
                cellLocation = self.mesh.ds.cellLocation
                np.savez(meshfilepath+f"mesh{i+1}.npz",node=newnode,cell=cell,cellLocation=cellLocation)
        self.errorMatrix = errorMatrix
        self.maxit = maxit
        
    def showresult(self,select_number=4):
        '''
        展示求解结果

        Parameters:
            select_number: 选取最后几次迭代的结果次数计算收敛阶，默认为4

        Returns:
            None
        '''

        errorType = ['$|| u - \Pi u_h||_{\Omega,0}$',
         '$||\\nabla u - \Pi \\nabla u_h||_{\Omega, 0}$',
         '$\eta $']

        from ..tools.show import showmultirate
        showmultirate(plt, self.maxit-select_number, np.array(self.NDof), self.errorMatrix, errorType, propsize=20, lw=2, ms=4)
        print(self.errorMatrix)
        plt.show()
        fig1 = plt.figure()
        axes  = fig1.gca()
        self.mesh.add_plot(axes)
        plt.show()

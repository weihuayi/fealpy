import numpy as np
import os
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator, BilinearForm, ScalarMassIntegrator, LinearForm, ScalarSourceIntegrator
from fealpy.fem.dirichlet_bc import DirichletBC

class ParabolicData:
    def domain(self):
        return [0, 10, 0, 10]
    
    def duration(self):
        return [0, 10]
    
    def source(self):
        return 0 
     
    def dirichlet(self, p):
        
        return np.array([500])

class HeatEquationSolver:
    def __init__(self, mesh: TriangleMesh, pde: ParabolicData, nt, bdnidx, p0, alpha_caldding=4e-4, alpha_inner=8e-4, layered=True, ficdx=None ,cacidx=None,output: str = './result', filename: str = 'temp'):
        """
        Args:
            mesh (TriangleMesh): 三角形网格
            pde (ParabolicData): 双曲方程的数据
            nt (_type_): 时间迭代步长
            bdnidx (_type_): 边界条件，布尔值or索引编号
            p0 (_type_): 初始温度
            layered (bool, optional):刚度矩阵分层. Defaults to True.
            output (str, optional): 生成文件的名字. Defaults to './result'.
            filename (str, optional): Defaults to 'temp'.
        """
        self.mesh = mesh
        self.pde = pde
        self.bdnidx = bdnidx
        self.layered = layered
        self.output = output
        self.filename = filename
        self.space = LagrangeFESpace(mesh, p=1)
        self.duration = pde.duration()
        self.nt = nt
        self.p0 = p0
        self.ficdx=ficdx
        self.cacidx=cacidx
        self.tau = (self.duration[1] - self.duration[0]) / self.nt
        self.alpha_caldding = alpha_caldding
        self.alpha_inner = alpha_inner
        self.initialize_output_directory()
        self.assemble_matrices()
        self.threshold = self.create_threshold()

    def initialize_output_directory(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def create_threshold(self):
        """
        可以分辨布尔数组和编号索引，如果是布尔数组直接传入，如果是索引编号转换为布尔数组
        """
        if isinstance(self.bdnidx, np.ndarray) and self.bdnidx.dtype == bool:
            return self.bdnidx
        else:
            NN = len(self.mesh.entity('node'))
            isbdnidx = np.full(NN, False, dtype=bool)
            isbdnidx[self.bdnidx] = True
            return isbdnidx

    def assemble_matrices(self):
        # 全局矩阵组装
        source = self.pde.source()
        bform3 = LinearForm(self.space)
        bform3.add_domain_integrator(ScalarSourceIntegrator(source, q=3))
        self.F = bform3.assembly()

        # 组装刚度矩阵
        NC = self.mesh.number_of_cells()
        alpha = np.zeros(NC)
        
        if self.layered:
            # 假设 ficdx 和 cacidx 是定义好的两个索引列表
            # 默认分层
            alpha[self.ficdx] += self.alpha_inner
            alpha[self.cacidx] += self.alpha_caldding
        else:
            # 如果不分层，使用统一的 alpha_caldding
            alpha += self.alpha_caldding

        bform = BilinearForm(self.space)
        bform.add_domain_integrator(DiffusionIntegrator(alpha, q=3))
        self.K = bform.assembly()

        # 组装质量矩阵
        bform2 = BilinearForm(self.space)
        bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
        self.M = bform2.assembly()

        self.p = np.zeros_like(self.F)
        self.p += self.p0

    def apply_boundary_conditions(self, A, b, t):
        bc = DirichletBC(space=self.space, gD=self.pde.dirichlet, threshold=self.threshold)
        A, b = bc.apply(A, b)
        return A, b

    def solve(self):
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            A = self.M + self.alpha_caldding * self.K * self.tau
            b = self.M @ self.p + self.tau * self.F
            if n == 0:
                A = A
                b = b
            else:
                A, b = self.apply_boundary_conditions(A, b, t)
            self.p = spsolve(A, b)
            self.mesh.nodedata['temp'] = self.p.flatten('F')
            name = os.path.join(self.output, f'{self.filename}_{n:010}.vtu')
            self.mesh.to_vtk(fname=name)
        print(self.p)
        print(self.p.shape)


# 使用示例:三维的燃料棒slover
if __name__ == "__main__":
    mm = 1e-03
    #包壳厚度
    w = 0.15 * mm
    #半圆半径
    R1 = 0.5 * mm
    #四分之一圆半径
    R2 = 1.0 * mm
    #连接处直线段
    L = 0.575 * mm
    #内部单元大小
    h = 0.5 * mm
    #棒长
    l = 20 * mm
    #螺距
    p = 40 * mm

from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher 
mesher = FuelRodMesher(R1,R2,L,w,h,l,p,meshtype='segmented',modeltype='3D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_3D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_3D_cnidx_bdnidx()
pde = ParabolicData()
FuelRodsolver = HeatEquationSolver(mesh, pde,64, bdnidx,300,ficdx=ficdx,cacidx=cacidx,output='./result_fuelrod3Dtest')
FuelRodsolver.solve()


"""
#使用示例，三维箱子热传导
if __name__ == "__main__":
    nx = 20
    ny = 20
    nz = 20
from fealpy.mesh import TetrahedronMesh
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1],nx,ny,nz)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
pde = ParabolicData()
Boxslover = HeatEquationSolver(mesh,pde,640,isBdNode,300,alpha_caldding=1.98e-2,layered=False,output='./rusult_boxtest')
Boxslover.solve()
"""



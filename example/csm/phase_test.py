import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, spdiags

from fealpy.mesh import TriangleMesh 
from fealpy.geometry import SquareWithCircleHoleDomain

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import ProvidesSymmetricTangentOperatorIntegrator

from fealpy.fem import DirichletBC

from mgis import behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

class Brittle_Facture_model():
    def __init__(self):
        self.E = 210 # 杨氏模量
        self.nu = 0.3 # 泊松比
        self.Gc = 1 # 材料的临界能量释放率
        self.l0 = 0.02 # 尺度参数，断裂裂纹的宽度

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.nu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量

    def top_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.concatenate((np.linspace(0, 70e-3, 6)[1:], np.linspace(70e-3,
            125e-3, 26)[1:]))

    def top_disp_direction(self):
        """
        @brief 上边界位移的方向
        Notes
        -----
        位移方向沿 (0, 1) 方向，即仅在 y 方向的位移变化
        """
        return np.array([0, 1], np.float_)

    def is_top_boundary(self, p):
        """
        @brief 标记上边界, y = 1 时的边界点
        """
        return np.abs(p[..., 1] - 1) < 1e-12 

    def is_inter_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001
    
    def is_below_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return np.abs(p[..., 1]) < 1e-12

class fracture_damage_integrator():
    def __init__(self, mesh, model):
        self.la = model.lam
        self.mu = model.mu
        self.E = model.E
        self.nu = model.nu
        self.ka = model.Gc
        self.l0 = model.l0
        self.mesh = mesh
        self._h = mgis_bv.Hypothesis.PlaneStrain # 平面
    
    def build_disp_mfront(self):
        file = compile_mfront_file('material/PhaseFieldDisplacementSpectralSplit.mfront')

        lib = "./libPhaseFieldDisplacementSpectralSplit.so"  # 用实际路径替换
        h = self._h

        # 加载行为模型
        self._b0 = mgis_bv.load(lib, "PhaseFieldDisplacementSpectralSplit", h)

    def update_disp_mfront(self, uh, H, d):
        b0 = self._b0
        NC = self.mesh.number_of_cells()
        m0 = mgis_bv.MaterialDataManager(b0, NC)
        self._m0 = m0

        # 设置材料属性
        mgis_bv.setMaterialProperty(m0.s1, "YoungModulus", self.E) # 设置材料属性
        mgis_bv.setMaterialProperty(m0.s1, "PoissonRatio", self.nu)
        mgis_bv.setExternalStateVariable(m0.s1, "Temperature", 293.15) #设置外部状态变量
        # 给定相场值
        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        mgis_bv.setExternalStateVariable(m0.s1, "Damage", d(bc),
                mgis_bv.MaterialStateManagerStorageMode.EXTERNAL_STORAGE) #设置外部状态变量

        # 初始化局部变量
        mgis_bv.update(m0) # 更新材料数据
        eto = self.strain(uh)
        m0.s0.internal_state_variables[:, 0] = H
        m0.s1.gradients[:] = eto

    def disp_tangent_matrix(self):
        m0 = self._m0
        it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
        dt = 0.0 
        mgis_bv.integrate(m0, it, dt, 0, m0.n)
        M = m0.K
        print('MMM:', M)
        #将切算子矩阵转化为felapy中默认的储存顺序
        M = np.delete(M, 2, axis=1)
        M = np.delete(M, 2, axis=2)
        M[:, -1, -1] = M[:, -1, -1]/2
        print('MMM:', M)
        return M
    
    def history_function(self):
        m0 = self._m0
        H = m0.s1.internal_state_variables[:, 0]
        return H

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        
        ____
        Notes:
        mfront 二维平面应变的默认保存顺序为[0, 0] [1, 1] 未知 [1, 0]
        ___
        """
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        gphi = self.mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 4), dtype=np.float64)
        uh = uh.T
        s[:, 0] = np.sum(uh[:, 0][cell] * gphi[:, :, 0], axis=-1)
        s[:, 1] = np.sum(uh[:, 1][cell] * gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell] * gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell] * gphi[:, :, 0], axis=-1)
        val /= 2.0
        s[:, 3] = val
        s[:, 2] = 0
        return s

    def build_damage_mfront(self):
        file = compile_mfront_file('material/PhaseFieldDamage.mfront')

        lib = "./libPhaseFieldDamage.so"  # 用实际路径替换
        h = self._h

        # 加载行为模型
        self._b1 = mgis_bv.load(lib, "PhaseFieldDamage", h)

    def update_damage_mfront(self, H, d):
        b1 = self._b1
        NC = self.mesh.number_of_cells()
        m1 = mgis_bv.MaterialDataManager(b1, NC)
        self._m1 = m1

        # 设置材料属性
        mgis_bv.setMaterialProperty(m1.s1, "RegularizationLength", self.l0) # 设置材料属性
        mgis_bv.setMaterialProperty(m1.s1, "FractureEnergy", self.ka)
        mgis_bv.setExternalStateVariable(m1.s1, "Temperature", 293.15) #设置外部状态变量
        mgis_bv.setExternalStateVariable(m1.s1, "HistoryFunction", H,
                mgis_bv.MaterialStateManagerStorageMode.LOCAL_STORAGE) #设置外部状态变量
        
        # 初始化局部变量
        mgis_bv.update(m1) # 更新材料数据
        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        g = d.grad_value(bc)
        m1.s1.gradients[:, 0] = d(bc)
        m1.s1.gradients[:, 1:] = g
    
    def damage_tangent_matrix(self):
        m1 = self._m1
        it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
        dt = 0.0 
        mgis_bv.integrate(m1, it, dt, 0, m1.n)
        M1 = m1.K
        print('DDDD:', M1)
        print('DDDD:', M1.shape)
        return M1
    

model = Brittle_Facture_model()

domain = SquareWithCircleHoleDomain() 
mesh = TriangleMesh.from_domain_distmesh(domain, 0.05, maxit=100)

GD = mesh.geo_dimension()
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()

simulation = fracture_damage_integrator(mesh, model)
simulation.build_disp_mfront()
simulation.build_damage_mfront()
space = LagrangeFESpace(mesh, p=1, doforder='sdofs')

d = space.function()
H = np.zeros(NC, dtype=np.float64)  # 分片常数
uh = space.function(dim=GD)
du = space.function(dim=GD)
dd = space.function()
disp = model.top_boundary_disp()
#for i in range(len(disp)):
node  = mesh.entity('node') 
isTNode = model.is_top_boundary(node)
uh[1, isTNode] = disp[0]
isTDof = np.r_['0', np.zeros(NN, dtype=np.bool_), isTNode]

#    k = 0
#    while k < 20:
#        print('i:', i)
#        print('k:', k)

# 位移
simulation.update_disp_mfront(uh, H, d)
D0 = simulation.disp_tangent_matrix()

vspace = (GD*(space, ))
ubform = BilinearForm(GD*(space, ))

integrator = ProvidesSymmetricTangentOperatorIntegrator(D0, q=4)
ubform.add_domain_integrator(integrator)
ubform.assembly()
A0 = ubform.get_matrix()
R0 = -A0@uh.flat[:]

ubc = DirichletBC(vspace, 0, threshold=model.is_inter_boundary)
A0, R0 = ubc.apply(A0, R0)

# 位移边界条件处理
bdIdx = np.zeros(A0.shape[0], dtype=np.int_)
bdIdx[isTDof] =1
Tbd =spdiags(bdIdx, 0, A0.shape[0], A0.shape[0])
T = spdiags(1-bdIdx, 0, A0.shape[0], A0.shape[0])
A0 = T@A0@T + Tbd
R0[isTDof] = du.flat[isTDof]

du.flat[:] = spsolve(A0, R0)
uh[:] += du

# 更新最大历史应变场
H = simulation.history_function()
print(H)
# 相场
simulation.update_damage_mfront(H, d)
D1 = simulation.damage_tangent_matrix()



fig = plt.figure()
axes = fig.add_subplot(111)
mesh.node += uh[:, :NN].T
mesh.add_plot(axes)
plt.show()


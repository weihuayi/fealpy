import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh 

from fealpy.geometry import SquareWithCircleHoleDomain
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm
from fealpy.fem import ProvidesSymmetricTangentOperatorIntegrator

import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

class ramberg_osgood_model():
    def __init__(self):
        self.E = 70000 # 杨氏模量
        self.nu = 0.2 # 泊松比
        self.n = 4.8
        self.alpha = 3/7
        self.sigma0 = 243 # 屈服应力 MPa

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.mu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数

    def right_boundary_force(self):
        """
        @brief 受力情况
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步力的大小
        """
        return np.concatenate((np.linspace(0, 133.65, 11)[1:], np.linspace(133.65,
            0, 11)[1:]))

    def right_force_direction(self):
        """
        @brief 右边界力的方向
        Notes
        -----
        位移方向沿 (1, 0) 方向，即向 x 的正向拉伸
        """
        return np.array([1, 0], np.float_)

    def is_top_boundary(self, p):
        """
        @brief 标记上边界, y = 1 时的边界点
        """
        return np.abs(p[..., 1] - 1) < 1e-12 

    def is_low_boundary(self, p):
        """
        @brief 标记下边界
        Notes
        -----
        DirichletBC，位移为 0
        """
        return np.abs(p[..., 0]) < 1e-12 

    def is_left_boundary(self, p):
        """
        @brief 标记左边界
        Notes
        -----
        DirichletBC，位移为 0
        """
        return np.abs(p[..., 1]) < 1e-12 

class ramberg_osgood_integrator():
    def __init__(self, mesh, model):
        self.E = model.E # 杨氏模量
        self.nu = model.nu # 泊松比
        self.n = model.n
        self.alpha = model.alpha
        self.sigma0 = model.sigma0 # 屈服应力 MPa
        self.mesh = mesh

    def build_mfront(self):
        file = compile_mfront_file('material/nonlinear_elasticity_ramberg_osgood_type.mfront')

        lib = "./libnonlinear_elasticity_ramberg_osgood_type.so"  
        h = mgis_bv.Hypothesis.PlaneStrain # 平面

        # 加载行为模型
        self._b = mgis_bv.load(lib, "RambergOsgoodNonLinearElasticity", h)

    def update_mfront(self, uh):
        b = self._b
        NC = self.mesh.number_of_cells()
        m = mgis_bv.MaterialDataManager(b, NC)
        self._m = m
        print(b.mps)

        mgis_bv.setMaterialProperty(m.s1, "YoungModulus", self.E) # 设置材料属性
        mgis_bv.setMaterialProperty(m.s1, "PoissonRatio", self.nu)
        mgis_bv.setMaterialProperty(m.s1, "n", self.n)
        mgis_bv.setMaterialProperty(m.s1, "alpha", self.alpha)
        mgis_bv.setMaterialProperty(m.s1, "YieldStrength", self.sigma0)
        mgis_bv.setExternalStateVariable(m.s1, "Temperature", 293.15)
        mgis_bv.update(m) # 更新材料数据
        eto = self.strain(uh)
        for i in range(NC):
            m.s1.gradients[i:] = eto[i:, :]

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        m = self._m
        it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
        dt = 0.1 
        mgis_bv.integrate(m, it, dt, 0, m.n)
        sigma = m.s1.thermodynamic_forces
        M = m.K

        #将切算子矩阵转化为felapy中默认的储存顺序
        M = np.delete(M, 1, axis=1)
        M = np.delete(M, 1, axis=2)
        M[:, [1, 2], :] = M[:, [2, 1], :]
        M[:, :, [1, 2]] = M[:, :, [2, 1]]
        return M

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None):    
        m = self.m
        sigma = m.s1.thermodynamic_forces
        
        #将应力转化为felapy中默认的储存顺序, 即二维情况盈利默认储存为 NC*3 的向量
        sigma = np.delete(sigma, 1, axis=1)
        sigma[:, [1, 2]] = sigma[:, [2, 1]]
        return sigma

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        
        ____
        Notes:
        mfront 二维平面应变的默认保存顺序为[0, 0] [0, 1] [1, 0] [1, 1]
        ___
        """
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        gphi = self.mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 4), dtype=np.float64)
        s[:, 0] = np.sum(uh[:, 0][cell] * gphi[:, :, 0], axis=-1)
        s[:, 3] = np.sum(uh[:, 1][cell] * gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell] * gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell] * gphi[:, :, 0], axis=-1)
        val /= 2.0
        s[:, 1] = val
        s[:, 2] = val
        return s

model = ramberg_osgood_model()

domain = SquareWithCircleHoleDomain() 
mesh = TriangleMesh.from_domain_distmesh(domain, 0.05, maxit=100)

GD = mesh.geo_dimension()
space = LagrangeFESpace(mesh, p=1, doforder='vdims')

u = space.function(dim=GD)

integrator = ramberg_osgood_integrator(mesh, model)
integrator.build_mfront()

bform = BilinearForm(GD*(space, ))

integrator.update_mfront(u)
bform.add_domain_integrator(integrator)
bform.assembly()

fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
plt.show()


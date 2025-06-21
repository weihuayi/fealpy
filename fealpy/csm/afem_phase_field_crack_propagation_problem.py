import numpy as np

from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm

from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarMassIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ProvidesSymmetricTangentOperatorIntegrator

from ..fem import DirichletBC
from ..fem import LinearRecoveryAlg
from ..mesh.adaptive_tools import mark

from scipy.sparse.linalg import spsolve

class AFEMPhaseFieldCrackPropagationProblem():
    """
    @brief 自适应线性有限元相场方法求解 2D 和 3D 裂纹传播问题
    """
    def __init__(self, model, mesh, p=1):
        """
        @brief 

        @param[in] model 算例模型
        @param[in] mesh 连续体离散网格
        @param[in] p 有限元空间次数
        """
        self.model = model
        self.GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        self.mesh = mesh
        self.space = LagrangeFESpace(mesh, p=p)

        self.uh = self.space.function(dim=GD) # 位移场
        self.d = self.space.function() # 相场
        self.H = np.zeros(NC) # 最大历史应变场
        self.recovery = LinearRecoveryAlg()

    def get_dissipated_energy(self, d):
        """
        @brief 计算耗散能量
        """
        model = self.model
        mesh = self.mesh

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        cm = mesh.entity_measure('cell')
        g = d.grad_value(bc)

        val = model.kappa/2/model.l0*(d(bc)**2+model.l0**2*np.sum(g*g, axis=1))
        dissipated = np.dot(val, cm)
        return dissipated

    
    def get_stored_energy(self, psi_s, d):
        """
        @brief  
        """
        eps = 1e-10
        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - d(bc)) ** 2 + eps
        mesh = self.mesh
        cm = mesh.entity_measure('cell')
        val = c0*psi_s
        stored = np.dot(val, cm)
        return stored

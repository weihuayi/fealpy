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

class AFEMPhaseFieldCrackPropagationProblem2d():
    def __init__(self, model, mesh, p=1):
        self.model = model
        self.mesh = mesh

        NC = mesh.number_of_cells()

        self.space = LagrangeFESpace(mesh, p=p)
        self.uh = self.space.function(dim=2)
        self.d = self.space.function()
        self.H = np.zeros(NC)

        # 初始化 self.uh

        self.index = np.array([
            (0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 1, 1),
            (0, 2, 0, 0, 0, 1),
            (1, 0, 1, 1, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (1, 2, 1, 1, 0, 1),
            (2, 0, 0, 1, 0, 0),
            (2, 1, 0, 1, 1, 1),
            (2, 2, 0, 1, 0, 1)], dtype=np.int_)


    def newton_raphson(self, ):
        pass

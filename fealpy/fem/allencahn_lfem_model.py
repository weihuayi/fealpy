from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEDataManager, ComputationalModel
from ..model.phasefield.ac_circle_data_2d import AcCircleData2D
from ..decorator import variantmethod,barycentric

# FEM imports
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import DirichletBC
from ..fem import (ScalarDiffusionIntegrator,
                   ScalarConvectionIntegrator,
                   ScalarMassIntegrator,
                   ScalarSourceIntegrator)
from ..solver import spsolve

class AllenCahnLFEMModel(ComputationalModel):
    """
    Allen-Cahn phase field model using Lagrange finite element method (LFEM).
    
    This model implements the Allen-Cahn equation in a weak form suitable for finite element analysis.
    It uses Lagrange finite element spaces for spatial discretization and supports time-stepping methods.
    """
    
    def __init__(self):
        super().__init__(pbar_log=True, log_level="INFO")
        self.fespace = LagrangeFESpace(self.data.geo_dimension(), 1)
        self.bilinear_form = BilinearForm(self.fespace)
        self.linear_form = LinearForm(self.fespace)
        self.dirichlet_bc = DirichletBC(self.fespace)
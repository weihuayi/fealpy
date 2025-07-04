from typing import Union
from ..backend import bm
from ..model import PDEDataManager, ComputationalModel
from ..model.surface_poisson import SurfaceLevelSetData
from ..decorator import variantmethod

# FEM import
from ..functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..sparse import COOTensor, CSRTensor

class SurfacePoissonLFEMModel(ComputationalModel):
    def __init__(self):
       self.pdm = PDEDataManager("surface_poisson")

    def set_pde(self):
        pass
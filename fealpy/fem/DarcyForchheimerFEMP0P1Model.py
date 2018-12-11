import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..solver import solve
from ..fem import doperator 
from .integral_alg import IntegralAlg

class DarcyForchheimerFEMP0P1Model:
    def __init__(self, pde, mesh):
        self.pde = pde
        self.
        

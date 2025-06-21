import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..solver import solve
from ..femmodel import doperator 
from .integral_alg import IntegralAlg

class DarcyForchheimerFEMModel:
    
    def __init__(self, pde, mesh):
        self.vspace = LagrangeFiniteElementSpace(mesh, p) 
        self.pspace = 
        self.mesh = self.femspace.mesh
        pass

    def solve(self):
        pass

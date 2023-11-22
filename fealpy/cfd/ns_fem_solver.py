import numpy as np
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm
from ..fem import LinearForm

class NSFEMSolver:
    def __init__(self, model, mesh, p=(2, 1)):
        self.model = model
        self.mesh = mesh
        self.uspace = LagrangeFESpace(mesh, p=p[0])
        self.pspace = LagrangeFESpace(mesh, p=p[1])

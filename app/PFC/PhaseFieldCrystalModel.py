import numpy as np
from scipy.sparse import bmat

from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import IsoLagrangeFiniteElementSpace

class PhaseFieldCrystalModel():
    def __init__(self, options):

        self.options = options
        p = options['order']
        mesh = options['mesh'] # Lagrange type mesh
        self.space = IsoLagrangeFiniteElementSpace(mesh, p)

        self.A = self.space.stiff_matrix()
        self.M = self.space.mass_matrix()

        self.ftype = mesh.ftype
        self.itype = mesh.itype

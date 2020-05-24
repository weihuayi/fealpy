import numpy as np


"""
This is a space on general convex quad and hex mesh
"""

class TensorProductFiniteElementSpace:
    def __init__(self, mesh, p, spacetype='C', q=None):
        self.p = p
        self.mesh = mesh

        self.GD = mesh.node.shape[1]

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

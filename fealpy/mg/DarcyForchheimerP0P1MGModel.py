import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..mesh import TriangleMesh

class DarchForchheimerP0P1MGModel:

    def __init__(self, pde, mesh, n):
        selp.pde = pde
        self.uspaces = []
        self.pspaces = []
        self.IMatrix = []

        mesh0 = TriangleMesh(mesh.node, mesh.ds.cell)
        uspace = VectorLagrangeFiniteElementSpace(mesh0, p=0, spacetype='D')
        self.uspaces.append(uspaces)
        pspace = LagrangeFiniteElementSpace(mesh0, p=1, spacetype='C')
        self.pspaces.append(pspace)

        for i in range(n):
            I0, I1 = mesh.uniform_refine()
            self.IMatrix.append((I0[0], I1[0]))
            mesh0 = TriangleMesh(mesh.node, mesh.ds.cell)
            uspace = VectorLagrangeFiniteElementSpace(mesh0, p=0, spacetype='D')
            self.uspaces.append(uspaces)
            pspace = LagrangeFiniteElementSpace(mesh0, p=1, spacetype='C')
            self.pspaces.append(pspace)

        self.uh = self.uspaces[-1].function()
        self.ph = self.pspaces[-1].function()

        self.uI = self.uspaces[-1].interpolation(pde.velocity)
        self.pI = self.pspaces[-1].interpolation(pde.pressure)

        self.nlevel = n + 1

    def compute_initial_value(self):
        pass


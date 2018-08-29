import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vector_vem_space import VectorVirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from .integral_alg import PolygonMeshIntegralAlg


class LinearElasticityVEMModel():
    def __init__(self, pde, mesh, p, q):
        self.pde = pde
        self.mesh = mesh
        self.space = VectorVirtualElementSpace2d(mesh, p)
        self.integrator = mesh.integrator(q)
        
        self.integralalg = PolygonMeshIntegralAlg(
           self.integrator, 
           mesh, 
           area=self.space.vsmspace.area, 
           barycenter=self.space.vsmspace.barycenter)

        self.uI = self.space.interpolation(pde.displacement, self.integralalg.integral)

    def matrix_G(self):
        mesh = self.mesh

        def f(x, cellidx): 
            sphi = self.space.vsmspace.strain_basis(x, cellidx=cellidx)
            dphi = self.space.vsmspace.div_basis(x, cellidx=cellidx)
            G = 2*np.einsum('ijkmn, ijpmn->ijkp', sphi, sphi)
            G += np.einsum('ijk, ijp->ijkp', dphi, dphi)
            return G 

        G = self.integralalg.integral(f, celltype=True);
        return G

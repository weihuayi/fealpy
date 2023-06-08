from fealpy.functionspace import ConformingScalarVESpace2d

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace import ConformingScalarVESpace2d
from .conforming_scalar_vem_h1_projector import ConformingScalarVEMH1Projector2d 
from .conforming_scalar_vem_l2_projector import ConformingScalarVEML2Projector2d

class ConformingScalarVEMLaplaceIntegrator():
    def __init__(self, space: ConformingScalarVESpace2d):
        self.space = space

    def assembly_cell_matrix(self, cfun=None):
        space = self.space
        p = space.p
        mesh = space.mesh


        H1Projector = ConformingScalarVEMH1Projector2d()
        G = H1Projector.assembly_cell_lefthand_side(space)
        D = H1Projector.assembly_cell_dof_matrix(space) 
        PI1 = H1Projector.assembly_cell_matrix(space)


        area = space.smspace.cellmeasure
        NC = mesh.number_of_cells()
        cell2dof = space.cell_to_dof() 

        def f(x):
            x[0, :] = 0
            return x

        tG = list(map(f, G))
        if cfun is None:
            f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
            K = list(map(f1, zip(D, PI1, tG)))
        else:
            pass
 
        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = np.concatenate(list(map(f2, cell2dof)))
        J = np.concatenate(list(map(f3, cell2dof)))
        val = np.concatenate(list(map(f4, K)))
        gdof = space.number_of_global_dofs()
        A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
        return A
    
    def source_vector(self, f):
        space = self.space
        L2project = ConformingScalarVEML2Projector2d()
        PI0 = L2project.assembly_cell_matrix(space)
        phi = space.smspace.basis
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        bb = space.integralalg.integral(u, celltype=True)
        g = lambda x: x[0].T@x[1]
        bb = np.concatenate(list(map(g, zip(PI0, bb))))
        gdof = space.number_of_global_dofs()
        b = np.bincount(np.concatenate(space.dof.cell2dof), weights=bb, minlength=gdof)
        return b






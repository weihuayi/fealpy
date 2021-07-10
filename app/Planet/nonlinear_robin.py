import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat

from fealpy.functionspace import WedgeLagrangeFiniteElementSpace

class nonlinear_robin():
    def __init__(self, pde, space, mesh, p=1, q=None, spacetype='c'):
        self.space = space
        self.mesh = mesh
        self.pde = pde
        self.p = p

    def robin_bc(self, A, uh, gR, threshold=None, q=None):
        p = self.p
        mesh = self.mesh

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = mesh.ds.boundary_tri_face_index()
            if threshold is not None:
                bc = mesh.entity_barycenter('face', ftype='tri', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.space.tri_face_to_dof()[index]

        qf0, qf1 = self.space.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf0.get_quadrature_points_and_weights()

        measure = mesh.boundary_tri_face_area(index=index)

        phi = self.space.basis(bcs)
        pp = mesh.bc_to_point(bcs, etype='face', ftype='tri', index=index)
        n = mesh.boundary_tri_face_unit_normal(bcs, index=index)
        
        if uh.coordtype == 'cartesian':
            uhval = uh(pp)
        elif uh.coordtype == 'barycentric':
            val = uh(bcs)
            uhval = val[:, index]

        val, kappa = gR(pp, n) # (NQ, NF, ...)

        phi0 = uhval[..., None]**3*phi

        FM = np.einsum('m, mi, mij, mik, i->ijk', ws, kappa, phi0, phi, measure)

        I = np.broadcast_to(face2dof[:, :, None], shape=FM.shape)
        J = np.broadcast_to(face2dof[:, None, :], shape=FM.shape)

        A+= csr_matrix((FM.flat, (I.flat, J.flat)), shape=A.shape)
        return A

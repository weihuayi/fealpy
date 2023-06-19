import numpy as np


class ScalarRobinBoundaryIntegrator:

    def __init__(self, kappa, q=3, threshold=None):
        self.kappa = kappa
        self.q = q
        self.threshold = threshold

    def assembly_face_matrix(self, space, out=None):

        q = self.q
        threshold = self.threshold
        kappa = self.kappa
        mesh = space.mesh
        gdof = space.number_of_global_dofs()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof(index)

        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        measure = mesh.entity_measure('face', index=index)

        phi = self.face_basis(bcs)
        pp = mesh.bc_to_point(bcs, index=index)
        n = mesh.face_unit_normal(index=index)

        val, kappa = gR(pp, n) # (NQ, NF, ...)

        bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)

        if len(val.shape) == 2:
            dim = 1
            if F is None:
                F = np.zeros((gdof, ), dtype=bb.dtype)
        else:
            dim = val.shape[-1]
            if F is None:
                F = np.zeros((gdof, dim), dtype=bb.dtype)

        if dim == 1:
            np.add.at(F, face2dof, bb)
        else:
            np.add.at(F, (face2dof, np.s_[:]), bb)

        FM = np.einsum('m, mi, mij, mik, i->ijk', ws, kappa, phi, phi, measure)
        I = np.broadcast_to(face2dof[:, :, None], shape=FM.shape)
        J = np.broadcast_to(face2dof[:, None, :], shape=FM.shape)
        R = csr_matrix((FM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        return R, F


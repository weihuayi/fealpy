import numpy as np


class ScalarNeumannBoundaryIntegrator:
    def __init__(self, space, gN, threshold=None, q=None):
        self.space = space
        self.gN = gN
        self.q = q
        self.threshold = threshold

    def assembly_face_vector(self, space, index=np.s_[:], facemeasure=None,
            out=None):
        """
        """
        gN = self.gN
        threshold = self.threshold
        mesh = space.mesh
        gdof = space.number_of_global_dofs()
       
        if isinstance(threshold, np.ndarray):
            index = threshold
        else:
            index = mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = space.face_to_dof(index=index)
        n = mesh.face_unit_normal(index=index)
        if facemeasure is None:
            facemeasure = mesh.entity_measure('face', index=index)

        q = self.q if self.q is not None else space.p + 1
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.face_basis(bcs)
        pp = mesh.bc_to_point(bcs, index=index)
        val = gN(pp, n) 

        if len(val.shape) == 2:
            dim = 1
            if F is None:
                F = np.zeros((gdof, ), dtype=self.ftype)
        else:
            dim = val.shape[-1]
            if F is None:
                F = np.zeros((gdof, dim), dtype=self.ftype)


        bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)
        if dim == 1:
            np.add.at(F, face2dof, bb)
        else:
            np.add.at(F, (face2dof, np.s_[:]), bb)

        return F

import numpy as np


class ScalarNeumannBoundaryIntegrator:
    def __init__(self, space, gN, threshold=None, q=None):
        self.space = space
        self.gN = gN
        self.q = None
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

        q = q if q is not None else space.p + 3 #TODO: 积分精度选择策略
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs)
        
        if callable(gN):
            if hasattr(gN, 'coordtype'):
                if gN.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    val = gN(ps, n) 
                elif gN.coordtype == 'barycentric':
                    val = gN(bcs, n, index=index)
            else: # 默认是笛卡尔
                ps = mesh.bc_to_point(bcs, index=index)
                val = gN(ps, n)
        else:
            val = gN


        if out is None:
            F = np.zeros((gdof, ), dtype=self.ftype)
        else:
            assert out.shape == (gdof,)
            F = out
        
        bb = np.einsum('q, qf, qfi, f->fi', ws, val, phi, facemeasure,
                optimize=True)
        np.add.at(F, face2dof, bb)
        
        if out is None:
            return F

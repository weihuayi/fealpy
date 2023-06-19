import numpy as np

class ScalarBoundarySourceIntegrator:
    def __init__(self, source, q=3, threshold=None):
        self.source = source 
        self.q = q 
        self.threshold = threshold

    def assembly_face_vector(self, space, out=None):
        """
        """
        q = self.q
        source = self.source
        threshold = self.threshold
        mesh = space.mesh
        gdof = space.number_of_global_dofs()
       
        if isinstance(threshold, np.ndarray):
            index = threshold
        else:
            index = mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]

        face2dof = space.face_to_dof(index=index)
        n = mesh.face_unit_normal(index=index)
        facemeasure = mesh.entity_measure('face', index=index)

        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs)
        
        if callable(source):
            if ~hasattr(source, 'coordtype') or source.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs, index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算 Neumann 边界的法向梯度
                val = source(ps, n) 
            elif source.coordtype == 'barycentric':
                # 这个时候 gN 是一个有限元函数，一定不需要算面法向
                val = source(bcs, index=index)
        else:
            val = source 

        if out is None:
            F = np.zeros((gdof, ), dtype=self.ftype)
        else:
            assert out.shape == (gdof,)
            F = out
        bb = np.einsum('q, qf, qfi, f->fi', ws, val, phi, facemeasure, optimize=True)
        np.add.at(F, face2dof, bb)

        if out is None:
            return F
        

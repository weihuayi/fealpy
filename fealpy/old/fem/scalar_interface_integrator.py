import numpy as np

class ScalarInterfaceIntegrator:
    """
    @brief 
    """
    def __init__(self, gI, threshold=None, q=3):
        self.gI = gI 
        self.q = q 
        self.threshold = threshold

    def assembly_face_vector(self, space, out=None):
        """
        """
        q = self.q
        gI = self.gI
        threshold = self.threshold
        mesh = space.mesh
        gdof = space.number_of_global_dofs()
       
        if isinstance(threshold, np.ndarray) or isinstance(threshold, slice):
            # 在实际应用中，一般直接给出相应网格实体编号
            # threshold 有以下几种情况：
            # 1. 编号数组
            # 2. slice
            # 3. 逻辑数组
            index = threshold 
        else:
            index = mesh.ds.boundary_face_index()
            if callable(threshold): 
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)] # 通过边界的重心来判断

        face2dof = space.face_to_dof(index=index)
        facemeasure = mesh.entity_measure('face', index=index)

        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs)
        
        if out is None:
            F = np.zeros((gdof, ), dtype=mesh.ftype)
        else:
            assert out.shape == (gdof,)
            F = out

        if callable(gI):
            if ~hasattr(gI, 'coordtype') or gI.coordtype == 'cartesian':
                # n = mesh.face_unit_normal(index=index)
                ps = mesh.bc_to_point(bcs, index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算边界面上的法向梯度
                val = gI(ps) 
            elif gI.coordtype == 'barycentric':
                # 这个时候 source 是一个有限元函数，一定不需要算面法向
                # TODO：也许这里有问题
                val = gI(bcs, index=index)
        else:
            val = gI 

        if np.isscalar(val):
            bb = val*np.einsum('q, qfi, f->fi', ws, phi, facemeasure, optimize=True)
        elif val.shape == facemeasure.shape:
            bb = np.einsum('q, f, qfi, f->fi', ws, val, phi, facemeasure, optimize=True)
        else:
            bb = np.einsum('q, qf, qfi, f->fi', ws, val, phi, facemeasure, optimize=True)

        np.add.at(F, face2dof, bb)

        if out is None:
            return F
        

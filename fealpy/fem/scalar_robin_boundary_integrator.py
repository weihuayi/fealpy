import numpy as np
from scipy.sparse import csr_matrix


class ScalarRobinBoundaryIntegrator:
    """
    @brief 标量类型的 Robin 边界条件，本质是一个边界面上的质量矩阵
    <\\kappa u, v> 
    """

    def __init__(self, kappa, threshold=None, q=3):
        """
        @param[in] kappa 一个标量值

        @todo 考虑 kappa 为变系数的情况
        """
        self.kappa = kappa
        self.q = q
        self.threshold = threshold

    def assembly_face_matrix(self, space, out=None):
        """
        """
        q = self.q
        threshold = self.threshold
        kappa = self.kappa
        mesh = space.mesh
        gdof = space.number_of_global_dofs()

        if isinstance(threshold, np.ndarray) or isinstance(threshold, slice):
            # 在实际应用中，一般直接给出相应网格实体编号
            # threshold 有以下几种情况：
            # 1. 编号数组
            # 2. slice
            # 3. 逻辑数组
            # TODO: 更新其它地方的 threshold
            index = threshold 
        else:
            index = mesh.ds.boundary_face_index()
            if callable(threshold): 
                bc = mesh.entity_barycenter('face', index=index)
                flag = threshold(bc) # 通过边界的重心来判断
                index = index[flag]

        face2dof = space.face_to_dof(index)

        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        measure = mesh.entity_measure('face', index=index)

        phi = space.face_basis(bcs)
        pp = mesh.bc_to_point(bcs, index=index)
        #TODO: 可能需要加入一个备用参数 bcs 
        # 如果 face 是曲，应该是每个积分点处有一个法向
        n = mesh.face_unit_normal(index=index) 

        FM = kappa*np.einsum('q, qci, qcj, c->cij', ws, phi, phi, measure, optimize=True)
        I = np.broadcast_to(face2dof[:, :, None], shape=FM.shape)
        J = np.broadcast_to(face2dof[:, None, :], shape=FM.shape)
        R = csr_matrix((FM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        if out is None:
            return R
        else:
            out += R


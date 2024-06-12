import fealpy.functionspace
import numpy as np


class PotentialGradPotentialIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_face_matrix(self, bd_space, space:fealpy.functionspace.LagrangeFESpace=None):
        """
        @note 没有参考单元的组装方式
        """
        coef = self.coef
        q = self.q
        if not isinstance(bd_space, tuple):
            space0 = bd_space
        else:
            GD = len(bd_space)
            space0 = bd_space[0]

        if space is None:
            space1 = space0
            index = np.s_[:]
        else:
            space1 = space
            index = ~space1.is_boundary_dof()


        bd_mesh = space0.mesh
        bd_node = bd_mesh.entity('node')
        bd_cell = bd_mesh.entity('cell')
        bd_cell_measure = bd_mesh.entity_measure('cell')

        GD = space1.GD
        TD = space1.TD
        ldof = space1.number_of_local_dofs()

        mesh = space1.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        cell_measure = mesh.entity_measure('cell')

        # 获取整体边界网格节点坐标
        # bd_gdof * dim

        cell2dof = space1.dof.cell_to_dof()
        if space0.p == 0:
            gdof = cell.shape[0]
            mul_idx = 0.5*np.ones((1, cell.shape[-1]))
            cell_point = np.einsum('cid,oi->cod', node[cell], mul_idx)
        else:
            gdof = space1.number_of_global_dofs()
            mul_idx = mesh.multi_index_matrix(space1.p, TD)
            cell_point = np.einsum('cid,oi->cod', node[cell], mul_idx / space1.p)
        xi = np.zeros((gdof, GD))
        xi[cell2dof] = cell_point
        xi = xi[index]
        # 每个面的两个节点
        x1 = bd_node[bd_cell[:, 0]]  # bd_NF * dim
        x2 = bd_node[bd_cell[:, 1]]

        qf = bd_mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        # xi = node
        ps = np.einsum('qj, ejd->qed', bcs, bd_node[bd_cell], optimize=True)

        phi = space0.basis(bcs)  # (NQ, NC, ldof, ...)
        # 相关数据计算
        # bd_gdof * bd_NF
        # c = np.sign((x1[np.newaxis, :, 0] - xi[..., np.newaxis, 0]) * (
        #         x2[np.newaxis, :, 1] - xi[..., np.newaxis, 1]) - (
        #                     x2[np.newaxis, :, 0] - xi[..., np.newaxis, 0]) * (
        #                     x1[np.newaxis, :, 1] - xi[..., np.newaxis, 1]))
        # h = c * np.abs((xi[..., np.newaxis, 0] - x1[np.newaxis, :, 0]) * (
        #         x2[np.newaxis, :, 1] - x1[np.newaxis, :, 1]) - (
        #                        xi[..., np.newaxis, 1] - x1[np.newaxis, :, 1]) * (
        #                        x2[np.newaxis, :, 0] - x1[np.newaxis, :, 0])) / bd_cell_measure[np.newaxis, :]

        # NF
        n = bd_mesh.cell_normal()
        h = np.einsum('fd, nfd -> nf', n, x1[np.newaxis, ...] - xi[..., np.newaxis, :]) / np.linalg.norm(n, axis=-1)

        # bd_gdof * NQ * bd_NF
        r = np.sqrt(np.sum((ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))

        # 单元自由度矩阵计算
        Hij = np.einsum('f, nf, q, qfi, nqf -> nfi', bd_cell_measure, h, ws, phi, 1 / r ** 2, optimize=True) / (
                    -2 * np.pi)
        Gij = np.einsum('f, q, qfi, nqf -> nfi', bd_cell_measure, ws, phi, np.log(1 / r), optimize=True) / (2 * np.pi)

        return Hij, Gij





import numpy as np


class PotentialIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_face_matrix(self, space, index=np.s_[:], cellmeasure=None):
        """
        @note 没有参考单元的组装方式
        """
        coef = self.coef
        q = self.q
        if not isinstance(space, tuple):
            space0 = space
        else:
            GD = len(space)
            space0 = space[0]

        mesh = space0.mesh
        GD = mesh.geo_dimension()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)
        NC = len(cellmeasure)
        ldof = space0.number_of_local_dofs()

        D = np.zeros((NC, ldof, ldof), dtype=space0.ftype)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi = space0.basis(bcs, index=index)  # (NQ, NC, ldof, ...)

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        # 每个面的两个节点
        x1 = node[cell[:, 0]]  # bd_NF * dim
        x2 = node[cell[:, 1]]
        # 获取整体边界网格节点坐标
        # bd_gdof * dim
        xi = node
        cell_measure = mesh.entity_measure('cell')
        ps = np.einsum('qj, ejd->qed', bcs, node[cell], optimize=True)

        # 相关数据计算
        # bd_gdof * bd_NF
        c = np.sign((x1[np.newaxis, :, 0] - xi[..., np.newaxis, 0]) * (
                x2[np.newaxis, :, 1] - xi[..., np.newaxis, 1]) - (
                            x2[np.newaxis, :, 0] - xi[..., np.newaxis, 0]) * (
                            x1[np.newaxis, :, 1] - xi[..., np.newaxis, 1]))
        h = c * np.abs((xi[..., np.newaxis, 0] - x1[np.newaxis, :, 0]) * (
                x2[np.newaxis, :, 1] - x1[np.newaxis, :, 1]) - (
                               xi[..., np.newaxis, 1] - x1[np.newaxis, :, 1]) * (
                               x2[np.newaxis, :, 0] - x1[np.newaxis, :, 0])) / cell_measure[np.newaxis, :]
        # bd_gdof * NQ * bd_NF
        r = np.sqrt(np.sum((ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))

        # 单元自由度矩阵计算
        Hij = np.einsum('f, nf, q, qfi, nqf -> nfi', cell_measure, h, ws, phi, 1 / r ** 2, optimize=True) / (
                    -2 * np.pi)
        Gij = np.einsum('f, q, qfi, nqf -> nfi', cell_measure, ws, phi, np.log(1 / r), optimize=True) / (2 * np.pi)

        return Hij, Gij





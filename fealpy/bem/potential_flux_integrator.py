import fealpy.functionspace
import numpy as np


class PotentialFluxIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_face_matrix(self, bd_space, xi: np.ndarray=None):
        space = bd_space
        mesh = space.mesh

        q = self.q
        GD = space.GD
        TD = space.TD
        ldof = space.number_of_local_dofs()

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        cell_measure = mesh.entity_measure('cell')
        NC = cell.shape[0]

        # 获取计算节点坐标
        # (bd_gdof, dim) or (len(xi), dim)
        if xi is None:
            if not hasattr(space, 'xi'):
                cell2dof = space.dof.cell_to_dof()
                if space.p == 0:
                    gdof = cell.shape[0]
                    mul_idx = np.ones((1, cell.shape[-1]))/cell.shape[-1]
                    cell_point = np.einsum('cid,oi->cod', node[cell], mul_idx)
                else:
                    gdof = space.number_of_global_dofs()
                    if type(mesh).__name__ == "QuadrangleMesh":
                        mul_idx = mesh.multi_index_matrix(space.p, TD-1)
                        mul_idx = np.einsum('ui, vj->uvij', mul_idx, mul_idx).reshape(-1, 4)[:, [0, 2, 3, 1]]
                    else:
                        mul_idx = mesh.multi_index_matrix(space.p, TD)
                    cell_point = np.einsum('cid,oi->cod', node[cell], mul_idx / space.p)
                xi = np.zeros((gdof, GD))
                xi[cell2dof] = cell_point
                space.xi = xi
            else:
                xi = space.xi

        # 获取每个边界面上任意一点坐标（此处使用单元重心）
        x_f = mesh.entity_barycenter('cell')
        # 获取积分权重并计算积分点坐标
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)
        # ps = np.einsum('qj, ejd->qed', bcs, node[cell], optimize=True)
        # 计算积分点对应的（局部）基函数值
        phi = space.basis(bcs)  # (NQ, NC, ldof, ...)
        # 相关数据计算
        # 计算计算节点到边界面的有向距离
        n = mesh.cell_normal()
        h = np.einsum('fd, nfd -> nf', n, x_f[np.newaxis, ...] - xi[..., np.newaxis, :]) / np.linalg.norm(n, axis=-1)

        # 计算计算节点到积分点的距离（无符号）
        # bd_gdof * NQ * bd_NF
        r = np.sqrt(np.sum((ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))

        # 单元自由度矩阵计算
        if GD == 2:
            H = np.einsum('f, nf, q, qfi, nqf -> nfi', cell_measure, h, ws, phi, 1 / r ** 2, optimize=True) / (
                        -2 * np.pi)
            G = np.einsum('f, q, qfi, nqf -> nfi', cell_measure, ws, phi, np.log(1 / r), optimize=True) / (2 * np.pi)
        elif GD == 3:
            H = np.einsum('f, nf, q, qfi, nqf -> nfi', cell_measure, h, ws, phi, 1 / r ** 3, optimize=True) / (
                    -4 * np.pi)
            G = np.einsum('f, q, qfi, nqf -> nfi', cell_measure, ws, phi, 1 / r, optimize=True) / (
                        4 * np.pi)
        else:
            raise ValueError("GD must be 2 or 3.")

        return H, G





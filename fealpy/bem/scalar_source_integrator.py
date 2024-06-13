import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union


class ScalarSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray], q=None):
        """
        @brief

        @param[in] f
        """
        self.f = f
        self.q = q
        self.vector = None

    def assembly_cell_vector(self,
                             cal_space,
                             dof_space,
                             cellmeasure=None,
                             out=None):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        if cal_space == dof_space:
            index = ~dof_space.is_boundary_dof()
        else:
            index = np.s_[:]
        f = self.f
        p = dof_space.p
        q = self.q
        q = p + 3 if q is None else q
        GD = dof_space.GD
        TD = dof_space.TD

        cal_mesh = cal_space.mesh
        if cellmeasure is None:
            cellmeasure = cal_mesh.entity_measure('cell', index=index)

        qf = cal_mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if callable(f):
            if hasattr(f, 'coordtype'):
                if f.coordtype == 'barycentric':
                    val = f(bcs, index=index)
                elif f.coordtype == 'cartesian':
                    ps = cal_mesh.bc_to_point(bcs, index=index)
                    val = f(ps)
            else:  # 默认是笛卡尔
                ps = cal_mesh.bc_to_point(bcs, index=index)
                val = f(ps)
        else:
            val = f

        # 获取自由度对应的节点坐标
        cell2dof = dof_space.dof.cell_to_dof()
        dof_mesh = dof_space.mesh
        dof_node = dof_mesh.entity('node')
        dof_cell = dof_mesh.entity('cell')
        if dof_space.p == 0:
            gdof = cell.shape[0]
            mul_idx = 0.5 * np.ones((1, dof_cell.shape[-1]))
            cell_point = np.einsum('cid,oi->cod', dof_node[dof_cell], mul_idx)
        else:
            gdof = dof_space.number_of_global_dofs()
            mul_idx = dof_mesh.multi_index_matrix(dof_space.p, TD)
            cell_point = np.einsum('cid,oi->cod', dof_node[dof_cell], mul_idx / dof_space.p)
        xi = np.zeros((gdof, GD))
        xi[cell2dof] = cell_point
        xi = xi[index]
        dof_space.xi = xi

        cell_r = np.sqrt(np.sum((ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))
        f = np.einsum('c,q,nqc,qc->n', cellmeasure, ws, np.log(1 / cell_r), val, optimize=True) / (2**(GD-1) * np.pi)

        return f




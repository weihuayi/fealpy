import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
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
                             bd_space,
                             xi=None):
        space = bd_space
        f = self.f
        p = space.p
        q = self.q
        q = p + 3 if q is None else q
        GD = space.GD
        TD = space.TD

        mesh = space.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        domain_mesh = space.domain_mesh
        cell_measure = domain_mesh.entity_measure('cell')
        NC = len(cell_measure)

        # 获取计算节点坐标
        # (bd_gdof, dim) or (len(xi), dim)
        if xi is None:
            if  not hasattr(space, 'xi'):
                cell2dof = space.dof.cell_to_dof()
                if space.p == 0:
                    gdof = cell.shape[0]
                    mul_idx = 0.5 * np.ones((1, cell.shape[-1]))
                    cell_point = np.einsum('cid,oi->cod', node[cell], mul_idx)
                else:
                    gdof = space.number_of_global_dofs()
                    mul_idx = mesh.multi_index_matrix(space.p, TD)
                    cell_point = np.einsum('cid,oi->cod', node[cell], mul_idx / space.p)
                xi = np.zeros((gdof, GD))
                xi[cell2dof] = cell_point
                space.xi = xi
            else:
                xi = space.xi

        # 获取积分权重并计算积分点坐标
        qf = domain_mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        if callable(f):
            if hasattr(f, 'coordtype'):
                if f.coordtype == 'barycentric':
                    val = f(bcs)
                elif f.coordtype == 'cartesian':
                    ps = domain_mesh.bc_to_point(bcs)
                    val = f(ps)
            else:  # 默认是笛卡尔
                ps = domain_mesh.bc_to_point(bcs)
                val = f(ps)
        else:
            val = f

        num_of_xi = len(xi)
        # r = np.sqrt(np.sum((ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))
        t = ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]
        r = np.linalg.norm(t, axis=-1)
        # r = np.zeros((len(xi),)+ps.shape[:-1])
        # for i, x in enumerate(xi):
        #     r[i] = np.linalg.norm(ps - x, axis=-1)
        # ps = np.broadcast_to(ps[np.newaxis, ...], (num_of_xi, len(ws), NC, GD)).reshape((-1, GD))
        # xi = np.broadcast_to(xi[:, np.newaxis, np.newaxis, ...], (num_of_xi, len(ws), NC, GD)).reshape((-1, GD))
        # r = cdist(ps, xi).reshape((num_of_xi, len(ws), NC))
        if GD == 2:
            f = np.einsum('c,q,nqc,qc->n', cell_measure, ws, np.log(1 / r), val, optimize=True) / (2**(GD-1) * np.pi)
        elif GD == 3:
            f = np.einsum('c,q,nqc,qc->n', cell_measure, ws, 1 / r, val, optimize=True) / (
                        2 ** (GD - 1) * np.pi)

        return f




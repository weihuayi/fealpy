"""
Common geometric / assembly utilities for metric-tensor mesh adaptivity.

Goal: share edge/metric construction, boundary projector, Jacobian sparsity
pattern, and simple assemblers between MetricTensorAdaptive and
MetricTensorAdaptiveX (or any new variant). Keep it backend-agnostic by using
fealpy.backend.backend_manager as bm.
"""
from dataclasses import dataclass
from typing import Optional
from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from scipy.sparse import coo_matrix, spdiags
import scipy.sparse as sp  # keep consistent with existing code style


@dataclass
class JacPattern:
    """Stores Jacobian sparsity pattern and index helpers."""
    I: TensorLike
    J: TensorLike
    rows_tile_d: TensorLike
    rows_base: TensorLike
    cols_x_tile_d: TensorLike
    cols_y_tile_d: TensorLike
    cols_x_0: TensorLike
    cols_y_0: TensorLike
    sparsity: Optional[sp.spmatrix]


@dataclass
class GeometricDiscreteCore:
    """
    OO-style facade wrapping the shared geometric/discrete utilities.

    This keeps MetricTensorAdaptive / MetricTensorAdaptiveX side effects
    (mesh, cm, kwargs0) in one place and exposes the same helpers as methods.
    """    
    def __init__(self, mesh):
        self.mesh = mesh
        self.kwargs0 = bm.context(mesh.node)

    def edge_matrix(self, X):
        """
        边矩阵 E
        E = [x_1 - x_0, x_2 - x_0, ..., x_d - x_0]
        """
        X0 = X[self.mesh.cell[:, 0], :]
        E = X[self.mesh.cell[:, 1:], :] - X0[:, None, :]
        return bm.permute_dims(E, axes=(0, 2, 1))

    def A(self, E_K, E_hat, M_inv):
        """
        基本的雅可比算子组件 A
        A = E_hat E_K^{-1}M^{-1} E_K^{-T} E_hat^T
        """
        E_K_inv = bm.linalg.inv(E_K)
        E_K_inv_T = bm.swapaxes(E_K_inv, -1, -2)
        E_hat_T = bm.swapaxes(E_hat, -1, -2)
        return E_hat @ E_K_inv @ M_inv @ E_K_inv_T @ E_hat_T

    def rho(self, M):
        """
        权重函数 rho , 一般形式
        rho = (det(M))^{1/2}
        Parameters
              M: (NC, GD, GD)
        """
        return bm.sqrt(bm.linalg.det(M))

    def theta(self, M):
        """
        参数 theta 的计算
        Parameters
            M: (NC, GD, GD)
        """
        d = self.mesh.geo_dimension()
        cm = self.mesh.entity_measure('cell')
        det_M = bm.linalg.det(M)
        rho_K = bm.sqrt(det_M)
        sigma = bm.sum(cm * rho_K, axis=0)
        area = bm.sum(cm, axis=0)
        return (sigma / area) ** (-2.0 / d)

    def balance(self, M_node, theta, power=None , mixed=True):
        d = self.mesh.geo_dimension()
        if power is None:
            power = -d / 2.0
        det_M_node = bm.linalg.det(M_node)
        if mixed:
            M_dim = (1.0 / theta * det_M_node ** (1.0 / d)) ** 0.5
        else:
            M_dim = det_M_node ** (1.0 / d)
        return M_dim ** power

    def R_matrix(self):
        localFace = self.mesh.localFace
        dim = localFace.shape[0]
        R = bm.zeros((dim, dim - 1), **self.kwargs0)
        R[0, :] = -1
        R[1:, :] = bm.eye(dim - 1, **self.kwargs0)
        return R

    def bd_projector(self, bd_idx, normals, vertices_idx=None):
        """
        构造按“变量列”右乘的边界投影矩阵系数（2D）
        全局变量排列为 [X(:,0); X(:,1)] （Fortran/列优先展开）
        列投影: [Jx Jy] @ [[Pxx Pxy],[Pyx Pyy]]
        其中 P** 为 NN×NN 的对角稀疏矩阵（非边界为单位/零，边界为对应系数）
        """
        NN = self.mesh.number_of_nodes()
        one = bm.ones(NN, **self.kwargs0)
        zero = bm.zeros(NN, **self.kwargs0)
        d_xx = one.copy(); d_yy = one.copy(); d_xy = zero.copy(); d_yx = zero.copy()

        nx, ny = normals[:, 0], normals[:, 1]
        d_xx = bm.set_at(d_xx, bd_idx, 1.0 - nx * nx)
        d_yy = bm.set_at(d_yy, bd_idx, 1.0 - ny * ny)
        d_xy = bm.set_at(d_xy, bd_idx, -nx * ny)
        d_yx = bm.set_at(d_yx, bd_idx, -ny * nx)

        if vertices_idx is not None:
            d_xx = bm.set_at(d_xx, vertices_idx, 0.0)
            d_yy = bm.set_at(d_yy, vertices_idx, 0.0)
            d_xy = bm.set_at(d_xy, vertices_idx, 0.0)
            d_yx = bm.set_at(d_yx, vertices_idx, 0.0)

        Rxx = spdiags(d_xx, 0, NN, NN, format='coo')
        Ryy = spdiags(d_yy, 0, NN, NN, format='coo')
        Rxy = spdiags(d_xy, 0, NN, NN, format='coo')
        Ryx = spdiags(d_yx, 0, NN, NN, format='coo')
        
        rxx = Rxx.diagonal()
        ryy = Ryy.diagonal()
        rxy = Rxy.diagonal()
        ryx = Ryx.diagonal()
        return {'Rxx': Rxx, 'Ryy': Ryy, 'Rxy': Rxy, 'Ryx': Ryx,
                'rxx': rxx, 'ryy': ryy,
                'rxy': rxy, 'ryx': ryx}
        
        
    def jac_pattern(self, cache_sparsity=True):
        cell = self.mesh.cell
        NN = self.mesh.number_of_nodes()
        d = self.mesh.geo_dimension()

        rows_base = cell.reshape(-1)  # (NC*(d+1),)
        cols_c = [bm.repeat(cell[:, c], d+1) for c in range(1, d+1)]  # d 个 (NC*(d+1),)
        cols_0 = bm.repeat(cell[:, 0], d+1)                           # (NC*(d+1),)

        # 行索引：对 c=1..d 拼接（同样的 rows 重复 d 次），以及 c=0 一份
        _rows_tile_d = bm.tile(rows_base, d)     # (d*NC*(d+1),)
        _rows_base   = rows_base                 # (NC*(d+1),)

        # 列索引：x/y 两组，c=1..d 和 c=0
        _cols_x_tile_d = bm.concat(cols_c, axis=0)        # (d*NC*(d+1),)
        _cols_y_tile_d = _cols_x_tile_d + NN
        _cols_x_0      = cols_0                           # (NC*(d+1),)
        _cols_y_0      = cols_0 + NN

        # 行偏移（vx 行: 0..NN-1；vy 行: NN..2NN-1）
        _row_off_x_tile_d = _rows_tile_d
        _row_off_y_tile_d = _rows_tile_d + NN
        _row_off_x_0      = _rows_base
        _row_off_y_0      = _rows_base + NN

        self.rr_x_all = _row_off_x_tile_d % NN
        self.rr_y_all = _row_off_y_tile_d % NN
        self.rr_x0 = _row_off_x_0 % NN
        self.rr_y0 = _row_off_y_0 % NN
        
        self.I = bm.concat([
            _row_off_x_tile_d, _row_off_y_tile_d,
            _row_off_x_0,      _row_off_y_0,
            _row_off_x_tile_d, _row_off_y_tile_d,
            _row_off_x_0,      _row_off_y_0,
        ], axis=0)
        self.J = bm.concat([
            _cols_x_tile_d, _cols_x_tile_d,
            _cols_x_0,      _cols_x_0,
            _cols_y_tile_d, _cols_y_tile_d,
            _cols_y_0,      _cols_y_0,
        ], axis=0)
        

    def assemble_vector(self, local_vector):
        GD = self.mesh.geo_dimension()
        cm = self.mesh.entity_measure('cell')
        global_vector = bm.zeros((self.mesh.number_of_nodes(), GD), dtype=bm.float64)
        global_vector = bm.index_add(global_vector, self.mesh.cell, cm[:, None, None] * local_vector)
        return global_vector

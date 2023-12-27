import numpy as np

from .mesh_base import Mesh
from .mesh_data_structure import Mesh2dDataStructure

class LagrangeMesh(Mesh):
    def __init__(self, node, cell, manifold=None, p=1):
        self.node = node

    def ref_cell_measure(self):
        raise NotImplementedError

    def ref_face_measure(self):
        raise NotImplementedError

    def vtk_cell_type(self, etype='cell'):
        raise NotImplementedError

    def number_of_corner_nodes(self):
        raise NotImplementedError

    def jacobi_matrix(self, bc, index=np.s_[:], return_grad=False):
        """
        @brief 计算参考单元 u 到实际实际单元 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """

        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

        entity = self.entity(etype=TD)[index]
        gphi = self.grad_shape_function(bc, index=index)
        J = np.einsum(
                'cin, ...cim->...cnm',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NQ,NC,ldof,TD)
        if return_grad is False:
            return J #(NQ,NC,GD,TD)
        else:
            return J, gphi

    def shape_function(self, bc, p=None, index=np.s_[:]):
        raise NotImplementedError

    def grad_shape_function(self, bc, p=None, index=np.s_[:]):
        raise NotImplementedError

    def shape_function_base(self, bc: NDArray, p: int =1, mi: NDArray=None):
        """
        @brief

        @param[in] bc
        """
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = self.multi_index_matrix(p, etype=TD)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., mi, idx], axis=-1)
        return phi

    def tensor_shape_function(self, bc, p: int =1,  mi: NDArray=None):
        """
        @brief 多个一维标量构造二维和三维的标量基函数

        @param[in] bc
        """
        if isinstance(bc, np.anarray): 
            return self.shape_function_base(bc, p, mi)
        elif isinstance(bc, tuple):
            GD = len(bc)
            phi = [self._shape_function(val, p=p) for val in bc]
            if GD == 2:
                ldof = (p+1)**GD
                return np.einsum('im, jn->ijmn', phi[0], phi[1]).reshape(-1, ldof)
            elif GD == 3:
                ldof = (p+1)**GD
                return np.einsum('im, jn, ko->ijkmno', phi[0], phi[1], phi[2]).reshape(-1, ldof)
            elif ValueError()
        else:
            raise TypeError('`bc` should be a tuple or ndarray!')

    def grad_tensor_shape_function_base(self, bc: NDArray, p: int =1, index=np.s_[:]):
        """
        @ berif 计算1D情形下形函数的张量梯度

        @param[in] bc 
        """
        if p == 1:
            return np.ones_like(bc) # 对于线性形函数，梯度是常数 1

        TD = bc.shape[-1] - 1
        mi = self.multi_index_matrix(p, etype=TD)

        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1] + (p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., mi, idx], axis=-1)

        # 计算梯度
        dphi_dbc = np.zeros_like(A)
        dphi_dbc[..., 1:, :] = p
        dphi_dbc[..., 1:, :] *= np.cumprod(A[..., mi, idx], axis=-1)[:, :, None] / A[..., mi, idx]

         # 沿着最后一个维度求和
        grad_phi_base = np.sum(dphi_dbc, axis=-1)
        return grad_phi_base

    def grad_tensor_shape_function(self, bc, p: int =1, index=np.np.s_[:]):
        raise NotImplementedError

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

    def tensor_shape_function_base(self, bc: NDArray, p: int =1, mi: NDArray=None):
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

    def tensor_shape_function(self, bc: NDArray, p: int =1,  mi: NDArray=None):
        """
        @brief 多个一维标量构造二维和三维的标量基函数

        @param[in] bc
        """
        if len(bc.shape) == 1:
            return self.tensor_shape_function_base(bc, p, mi)

        elif len(bc.shape) == 2:
            phi_1d_x = self.tensor_shape_function_base(bc[..., 0], p, mi[..., 0])
            phi_1d_y = self.tensor_shape_function_base(bc[..., 1], p, mi[..., 1])
            phi = np.einsum('im, jn -> ijmn', phi_1d_x, phi_1d_y)

            shape = phi.shape[:-2] + (-1, )
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
            return phi

        elif len(bc.shape) == 3:
            phi_1d_x = self.tensor_shape_function_base(bc[..., 0], p, mi[..., 0])
            phi_1d_y = self.tensor_shape_function_base(bc[..., 1], p, mi[..., 1])
            phi_1d_z = self.tensor_shape_function_base(bc[..., 2], p, mi[..., 2])
            phi = np.einsum('il, jm, kn -> ijklmn' , phi_1d_x, phi_1d_y, phi_1d_z)

            shape = phi.shape[:-3] + (-1, )
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
            return phi
        else:
            raise ValueError("Unsupport dimensin, only 1D, 2D, and 3D are supported.")

    def grad_tensor_function(self, bc: NDArray, p: int =1, index=np.s_[:]):
        raise NotImplementedError

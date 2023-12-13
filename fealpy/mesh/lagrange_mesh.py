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

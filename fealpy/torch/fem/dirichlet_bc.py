

from typing import Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch import Tensor

from ..functionspace.space import FunctionSpace


class DirichletBC():
    def __init__(self, space: FunctionSpace, gD: Union[Callable[..., Tensor], Tensor],
                 threshold: Optional[Callable] = None):
        r""""""
        self.space = space
        self.gD = gD
        self.threshold = threshold
        self.bctype = 'Dirichlet'

    def apply(self, A: Tensor, f: Tensor, uh: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """
        @brief Process Dirichlet boundary condition.

        @param[in] A: coefficient matrix
        @param[in] f: right-hand-size vector
        @param[in] uh: solution vector
        """
        gdof = self.space.number_of_global_dofs()
        GD = int(A.shape[0]//gdof)
        if uh is None:
            uh = self.space.function(dim=GD)

        return self.apply_for_other_space(A, f, uh)

    # TODO: finish this
    def apply_for_other_space(self, A: Tensor, f: Tensor, uh: Tensor) -> Tuple[Tensor, Tensor]:
        """
        @brief 处理基是向量函数的向量函数空间或标量函数空间的 Dirichlet 边界条件
        """
        space = self.space
        gD = self.gD
        isDDof = space.is_boundary_dof(threshold=self.threshold)
        isDDof = space.interpolate(gD, uh, dof_idx=isDDof) # isDDof.shape == uh.shape
        f = f - A@uh.reshape(-1) # 注意这里不修改外界 f 的值

        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof.reshape(-1)] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1

        f[isDDof.reshape(-1)] = uh[isDDof].reshape(-1)

        return A, f

    # def apply_for_vspace_with_scalar_basis(self, A, f, uh, dflag=None):
    #     """
    #     @brief 处理基由标量函数组合而成的向量函数空间的 Dirichlet 边界条件

    #     @param[in]

    #     """
    #     space = self.space
    #     assert isinstance(space, tuple) and not isinstance(space[0], tuple)

    #     gD = self.gD
    #     if dflag is None:
    #         dflag = space[0].boundary_interpolate(gD, uh, threshold=self.threshold)
    #     f = f - A@uh.flat # 注意这里不修改外界 f 的值

    #     bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    #     bdIdx[dflag.flat] = 1
    #     D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    #     D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    #     A = D0@A@D0 + D1
    #     f[dflag.flat] = uh.ravel()[dflag.flat]
    #     return A, f

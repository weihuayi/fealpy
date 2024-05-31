

from typing import Optional, Tuple, Callable, Union

import torch
from torch import Tensor
from torch.sparse import mm

from ..functionspace.space import FunctionSpace


class DirichletBC():
    """@brief"""
    def __init__(self, space: FunctionSpace, gD: Union[Callable[..., Tensor], Tensor],
                 threshold: Optional[Callable]=None,
                 batch_first: bool=True):
        self.space = space
        self.gD = gD
        self.threshold = threshold
        self.batch_first = batch_first
        self.bctype = 'Dirichlet'

    def apply(self, A: Tensor, f: Tensor, uh: Tensor) -> Tuple[Tensor, Tensor]:
        """
        @brief Process Dirichlet boundary condition. This is out-of-place for
        A and f, while in-place for uh.

        @param[in] A: coefficient matrix
        @param[in] f: right-hand-size vector
        @param[in] uh: boundary-interpolated solution vector

        @Returns:
            A: coefficient matrix
            f: right-hand-size vector
        """
        return self.apply_for_other_space(A, f, uh)

    # TODO: finish this
    def apply_for_other_space(self, A: Tensor, f: Tensor, uh: Tensor) -> Tuple[Tensor, Tensor]:
        """@brief 处理基是向量函数的向量函数空间或标量函数空间的 Dirichlet 边界条件"""
        # Make sure that A is in the COO format and f is dense.
        if not A.layout == torch.sparse_coo:
            raise ValueError('The layout of A must be torch.sparse_coo.')
        if not A.is_coalesced():
            raise RuntimeError('The A must be coalesced.')
        if f.is_sparse:
            raise ValueError('The layout of f must be torch.dense.')

        space = self.space
        gD = self.gD
        isDDof = space.is_boundary_dof(threshold=self.threshold) # on the same device as space
        DIM = -1 if self.batch_first else 0
        space.interpolate(gD, uh, dim=DIM, index=isDDof) # isDDof.shape == uh.shape

        if self.batch_first:
            uh = uh.transpose(0, 1)
            f = f.transpose(0, 1)

        if uh.ndim == 1:
            f = f - mm(A, uh.unsqueeze(-1)).squeeze(-1)
        elif uh.ndim == 2:
            f = f - mm(A, uh)
        else:
            raise ValueError('The dimension of uh must be 1 or 2.')

        # NOTE: Code in the numpy version:
        # ```
        # bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        # bdIdx[isDDof.reshape(-1)] = 1
        # D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        # D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        # A = D0@A@D0 + D1
        # ```

        new_values = torch.zeros_like(A.values(), requires_grad=False)
        indices = A.indices()
        IDX = isDDof[indices[0, :]] & (indices[1, :] == indices[0, :])
        new_values[IDX] = 1.0
        IDX = ~(isDDof[indices[0, :]] | isDDof[indices[1, :]])
        new_values[IDX] = A.values()[IDX]
        A = torch.sparse_coo_tensor(indices, new_values, A.size())
        A = A.coalesce()

        bdIdx = torch.zeros_like(isDDof, requires_grad=False)
        bdIdx[isDDof.reshape(-1)] = True
        if uh.ndim == 2:
            bdIdx = bdIdx.unsqueeze_(1)
        f = f * ~bdIdx + uh * bdIdx

        if self.batch_first:
            f = f.transpose(0, 1)

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

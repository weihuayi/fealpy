

from typing import Optional, Tuple, Callable, Union

import torch
from torch import Tensor
from torch.sparse import mm

from ..functionspace.space import FunctionSpace

CoefLike = Union[float, int, Tensor, Callable[..., Tensor]]


class DirichletBC():
    """Dirichlet boundary condition."""
    def __init__(self, space: FunctionSpace,
                 gd: Optional[CoefLike]=None,
                 *, threshold: Optional[Callable]=None, left: bool=True):
        self.space = space
        self.gd = gd
        self.threshold = threshold
        self.left = left
        self.bctype = 'Dirichlet'

        isDDof = space.is_boundary_dof(threshold=self.threshold) # on the same device as space
        self.is_boundary_dof = isDDof
        self.boundary_dof_index = torch.nonzero(isDDof, as_tuple=True)[0]
        self.gdof = space.number_of_global_dofs()

    def check_matrix(self, matrix: Tensor, /) -> Tensor:
        """Check if the input matrix is available for Dirichlet boundary condition.

        Args:
            matrix (Tensor): The left-hand-side matrix of the linear system.

        Raises:
            ValueError: When the layout is not torch.sparse_coo.
            RuntimeError: When the matrix is not coalesced.
            ValueError: When the matrix is not 2-dimensional.
            ValueError: When the matrix is not square.
            ValueError: When the matrix size does not match the gdof of the space.

        Returns:
            Tensor: The input matrix object.
        """
        if not matrix.layout == torch.sparse_coo:
            raise ValueError('The layout of matrix must be torch.sparse_coo.')
        if not matrix.is_coalesced():
            raise RuntimeError('The matrix must be coalesced.')
        if len(matrix.shape) != 2:
            raise ValueError('The matrix must be a 2-D sparse COO matrix.')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('The matrix must be a square matrix.')
        if matrix.shape[0] != self.gdof:
            raise ValueError('The matrix size must match the gdof of the space.')
        return matrix

    def check_vector(self, vector: Tensor, /) -> Tensor:
        """Check if the input vector is available for Dirichlet boundary conditions.

        Args:
            vector (Tensor): The right-hand-side vector of the linear system.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            Tensor: The input vector object.
        """
        if vector.is_sparse:
            raise ValueError('The layout of vector must be torch.dense.')
        if vector.ndim not in (1, 2):
            raise ValueError('The vector must be 1-D or 2-D.')
        if self.left:
            if vector.shape[-1] != self.gdof:
                raise ValueError('The vector size must match the gdof of the space.')
        else:
            if vector.shape[0] != self.gdof:
                raise ValueError('The vector size must match the gdof of the space.')
        return vector

    def apply(self, A: Tensor, f: Tensor, uh: Optional[Tensor]=None,
              gd: Optional[CoefLike]=None, *,
              check=True) -> Tuple[Tensor, Tensor]:
        """Apply Dirichlet boundary conditions.

        Args:
            A (Tensor): _description_
            f (Tensor): _description_
            uh (Tensor, optional): The solution uh Tensor. Boundary interpolation\
                will be done on `uh` if given, which is an **in-place** operation.\
                Defaults to None.
            gd (CoefLike, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor]: New adjusted `A` and `f`.
        """
        f = self.apply_vector(f, A, uh, gd, check=check)
        A = self.apply_matrix(A, check=check)
        return A, f

    def apply_matrix(self, matrix: Tensor, *, check=True) -> Tensor:
        """Apply Dirichlet boundary condition to left-hand-size matrix only.

        Args:
            matrix (Tensor): The original left-hand-size COO sparse matrix\
                of the linear system.
            check (bool, optional): Whether to check the matrix. Defaults to True.

        Returns:
            Tensor: New adjusted left-hand-size COO matrix.
        """
        # NOTE: Code in the numpy version:
        # ```
        # bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        # bdIdx[isDDof.reshape(-1)] = 1
        # D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        # D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        # A = D0@A@D0 + D1
        # ```
        # Here the adjustment is done by operating the sparse structure directly.
        isDDof = self.is_boundary_dof
        A = self.check_matrix(matrix) if check else matrix
        kwargs = {'dtype': A.dtype, "device": A.device}
        indices = A.indices()
        new_values = A.values().clone()
        IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
        new_values[IDX] = 0
        A = torch.sparse_coo_tensor(indices, new_values, A.size(), **kwargs)
        index, = torch.nonzero(isDDof, as_tuple=True) 
        one_values = torch.ones(len(index))
        one_indices = torch.stack([index,index], dim=0)
        A1 = torch.sparse_coo_tensor(one_indices, one_values, A.size(), **kwargs)
        A += A1 
        A = A.coalesce()
        
        return A

    def apply_vector(self, vector: Tensor, matrix: Tensor, uh: Optional[Tensor]=None,
                     gd: Optional[CoefLike]=None, *, check=True) -> Tensor:
        """Appy Dirichlet boundary contition to right-hand-size vector only.

        Args:
            vector (Tensor): The original right-hand-size vector.
            matrix (Tensor): The original COO sparse matrix.
            uh (Optional[Tensor]): The solution uh Tensor. Defuault to None.\
                See `DirichletBC.apply()` for more details.
            gd (Optional[CoefLike]): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): Whether to check the vector. Defaults to True.

        Raises:
            RuntimeError: If gd is `None` and no default gd exists.

        Returns:
            Tensor: New adjusted right-hand-size vector.
        """
        A = matrix
        f = self.check_vector(vector) if check else vector
        gd = self.gd if gd is None else gd
        if gd is None:
            raise RuntimeError("The boundary condition is None.")
        bd_idx = self.boundary_dof_index
        DIM = -1 if self.left else 0

        if uh is None:
            uh = torch.zeros_like(f)
        self.space.interpolate(gd, uh, dim=DIM, index=bd_idx) # isDDof.shape == uh.shape

        if uh.ndim == 1:
            f = f - mm(A, uh.unsqueeze(-1)).squeeze(-1)
            f[bd_idx] = uh[bd_idx]

        elif uh.ndim == 2:
            if self.left:
                uh = uh.transpose(0, 1)
                f = f.transpose(0, 1)
            f = f - mm(A, uh)
            f[bd_idx] = uh[bd_idx]
            if self.left:
                f = f.transpose(0, 1)
        else:
            raise ValueError('The dimension of uh must be 1 or 2.')

        return f


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

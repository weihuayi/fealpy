

from typing import Optional, Tuple, Callable, Union

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_dot_general

from ..functionspace.space import FunctionSpace
from ..utils import Array

CoefLike = Union[float, int, Array, Callable[..., Array]]


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
        self.boundary_dof_index = jnp.nonzero(isDDof)[0]
        self.gdof = space.number_of_global_dofs()

    def check_matrix(self, matrix: Array, /) -> Array:
        """Check if the input matrix is available for Dirichlet boundary condition.

        Args:
            matrix (Array): The left-hand-side matrix of the linear system.

        Raises:
            ValueError: When the layout is not torch.sparse_coo.
            RuntimeError: When the matrix is not coalesced.
            ValueError: When the matrix is not 2-dimensional.
            ValueError: When the matrix is not square.
            ValueError: When the matrix size does not match the gdof of the space.

        Returns:
            Array: The input matrix object.
        """
        if not isinstance(matrix, BCOO):
            raise ValueError('The layout of matrix must be torch.sparse_coo.')
        # if not matrix.is_coalesced():
        #     raise RuntimeError('The matrix must be coalesced.')
        if len(matrix.shape) != 2:
            raise ValueError('The matrix must be a 2-D sparse COO matrix.')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('The matrix must be a square matrix.')
        if matrix.shape[0] != self.gdof:
            raise ValueError('The matrix size must match the gdof of the space.')
        return matrix

    def check_vector(self, vector: Array, /) -> Array:
        """Check if the input vector is available for Dirichlet boundary conditions.

        Args:
            vector (Array): The right-hand-side vector of the linear system.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            Array: The input vector object.
        """
        # if not isinstance(vector, BCOO):
        #     raise ValueError('The layout of vector must be torch.dense.')
        if vector.ndim not in (1, 2):
            raise ValueError('The vector must be 1-D or 2-D.')
        if self.left:
            if vector.shape[-1] != self.gdof:
                raise ValueError('The vector size must match the gdof of the space.')
        else:
            if vector.shape[0] != self.gdof:
                raise ValueError('The vector size must match the gdof of the space.')
        return vector

    def apply(self, A: Array, f: Array, uh: Optional[Array]=None,
              gd: Optional[CoefLike]=None, *,
              check=True) -> Tuple[Array, Array]:
        """Apply Dirichlet boundary conditions.

        Args:
            A (Array): _description_
            f (Array): _description_
            uh (Array, optional): The solution uh Array. Boundary interpolation\
                will be done on `uh` if given, which is an **in-place** operation.\
                Defaults to None.
            gd (CoefLike, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[Array, Array]: New adjusted `A` and `f`.
        """
        f = self.apply_vector(f, A, uh, gd, check=check)
        A = self.apply_matrix(A, check=check)
        return A, f

    def apply_matrix(self, matrix: Array, *, check=True) -> Array:
        """Apply Dirichlet boundary condition to left-hand-size matrix only.

        Args:
            matrix (Array): The original left-hand-size COO sparse matrix\
                of the linear system.
            check (bool, optional): Whether to check the matrix. Defaults to True.

        Returns:
            Array: New adjusted left-hand-size COO matrix.
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
        new_values = jnp.zeros_like(A.data)
        indices = A.indices.T
        IDX = isDDof[indices[0, :]] & (indices[1, :] == indices[0, :])
        # print(new_values.shape, IDX.shape)
        new_values = new_values.at[IDX].set(1.0)
        IDX = ~(isDDof[indices[0, :]] | isDDof[indices[1, :]])
        new_values = new_values.at[IDX].set(A.data[IDX])
        A = BCOO((new_values, indices.T), shape=A.shape)

        return A

    def apply_vector(self, vector: Array, matrix: Array, uh: Optional[Array]=None,
                     gd: Optional[CoefLike]=None, *, check=True) -> Array:
        """Appy Dirichlet boundary contition to right-hand-size vector only.

        Args:
            vector (Array): The original right-hand-size vector.
            matrix (Array): The original COO sparse matrix.
            uh (Optional[Array]): The solution uh Array. Defuault to None.\
                See `DirichletBC.apply()` for more details.
            gd (Optional[CoefLike]): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): Whether to check the vector. Defaults to True.

        Raises:
            RuntimeError: If gd is `None` and no default gd exists.

        Returns:
            Array: New adjusted right-hand-size vector.
        """
        A = matrix
        f = self.check_vector(vector) if check else vector
        gd = self.gd if gd is None else gd
        if gd is None:
            raise RuntimeError("The boundary condition is None.")
        bd_idx = self.boundary_dof_index
        DIM = -1 if self.left else 0

        if uh is None:
            uh = jnp.zeros_like(f)
        self.space.interpolate(gd, uh, dim=DIM, index=bd_idx) # isDDof.shape == uh.shape

        if uh.ndim == 1:
            f = f - jnp.squeeze(bcoo_dot_general(A, jnp.expand_dims(uh, axis=-1), dimension_numbers=(((1,), (0,)), ((), ()))), axis=-1)
            
            f = f.at[bd_idx].set(uh[bd_idx])

        elif uh.ndim == 2:
            if self.left:
                uh = jnp.swapaxes(uh, 0, 1)
                f = jnp.swapaxes(f, 0, 1)
            f = f - jnp.matmul(A, uh)
            f = f.at[bd_idx].set(uh[bd_idx])
            if self.left:
                f = jnp.swapaxes(f, 0, 1)
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

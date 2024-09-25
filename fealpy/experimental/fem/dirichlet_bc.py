
from typing import Optional, Tuple, Callable, Union, TypeVar

from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import SparseTensor, COOTensor, CSRTensor
from ..functionspace.space import FunctionSpace

CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]
_ST = TypeVar('_ST', bound=SparseTensor)


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
        self.boundary_dof_index = bm.nonzero(isDDof)[0]
        self.gdof = space.number_of_global_dofs()

    def check_matrix(self, matrix: SparseTensor, /) -> SparseTensor:
        """Check if the input matrix is available for Dirichlet boundary condition.

        Parameters:
            matrix (COOTensor): The left-hand-side matrix of the linear system.

        Raises:
            ValueError: When the layout is not torch.sparse_coo.
            RuntimeError: When the matrix is not coalesced.
            ValueError: When the matrix is not 2-dimensional.
            ValueError: When the matrix is not square.
            ValueError: When the matrix size does not match the gdof of the space.

        Returns:
            Tensor: The input matrix object.
        """
        if not isinstance(matrix, (COOTensor, CSRTensor)):
            raise ValueError('The type of matrix must be COOTensor or CSRTensor.')
        if not matrix.is_coalesced:
            raise RuntimeError('The matrix must be coalesced.')
        if len(matrix.shape) != 2:
            raise ValueError('The matrix must be a 2-D sparse COO matrix.')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('The matrix must be a square matrix.')
        if matrix.shape[0] != self.gdof:
            raise ValueError('The matrix size must match the gdof of the space.')
        return matrix

    def check_vector(self, vector: TensorLike, /) -> TensorLike:
        """Check if the input vector is available for Dirichlet boundary conditions.

        Parameters:
            vector (Tensor): The right-hand-side vector of the linear system.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            Tensor: The input vector object.
        """
        if not bm.is_tensor(vector):
            raise ValueError('The type of vector must be a tensor.')
        if vector.ndim not in (1, 2):
            raise ValueError('The vector must be 1-D or 2-D.')
        if self.left:
            if vector.shape[-1] != self.gdof:
                raise ValueError('The vector size must match the gdof of the space.')
        else:
            if vector.shape[0] != self.gdof:
                raise ValueError('The vector size must match the gdof of the space.')
        return vector

    def apply(self, A: SparseTensor, f: TensorLike, uh: Optional[TensorLike]=None,
              gd: Optional[CoefLike]=None, *,
              check=True) -> Tuple[TensorLike, TensorLike]:
        """Apply Dirichlet boundary conditions.

        Parameters:
            A (COOTensor): Left-hand-size COO sparse matrix
            f (Tensor): Right-hand-size vector
            uh (Tensor | None, optional): The solution uh Tensor. Boundary interpolation\
                will be done on `uh` if given, which is an **in-place** operation.\
                Defaults to None.
            gd (CoefLike | None, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): _description_. Defaults to True.

        Returns:
            out (Tensor, Tensor): New adjusted `A` and `f`.
        """
        f = self.apply_vector(f, A, uh, gd, check=check)
        A = self.apply_matrix(A, check=check)
        return A, f

    def apply_matrix(self, matrix: _ST, *, check=True) -> _ST:
        """Apply Dirichlet boundary condition to left-hand-size matrix only.

        Parameters:
            matrix (COOTensor): The original left-hand-size COO sparse matrix\
                of the linear system.
            check (bool, optional): Whether to check the matrix. Defaults to True.

        Returns:
            COOTensor: New adjusted left-hand-size COO matrix.
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
        A = self.check_matrix(matrix) if check else matrix
        isDDof = self.is_boundary_dof
        kwargs = A.values_context()

        if isinstance(A, COOTensor):
            indices = A.indices()
            remove_flag = bm.logical_or(
                isDDof[indices[0, :]], isDDof[indices[1, :]]
            )
            retain_flag = bm.logical_not(remove_flag)
            new_indices = indices[:, retain_flag]
            new_values = A.values()[..., retain_flag]
            A = COOTensor(new_indices, new_values, A.sparse_shape)

            index = bm.nonzero(isDDof)[0]
            shape = new_values.shape[:-1] + (len(index), )
            one_values = bm.ones(shape, **kwargs)
            one_indices = bm.stack([index, index], axis=0)
            A1 = COOTensor(one_indices, one_values, A.sparse_shape)
            A = A.add(A1).coalesce()

        elif isinstance(A, CSRTensor):
            raise NotImplementedError('The CSRTensor version has not been implemented.')

        return A

    def apply_vector(self, vector: TensorLike, matrix: SparseTensor,
                     uh: Optional[TensorLike]=None,
                     gd: Optional[CoefLike]=None, *, check=True) -> TensorLike:
        """Appy Dirichlet boundary contition to right-hand-size vector only.

        Parameters:
            vector (TensorLike): The original right-hand-size vector.
            matrix (COOTensor): The original COO/CSR sparse matrix.
            uh (TensorLike | None, optional): The solution uh Tensor. Defuault to None.\
                See `DirichletBC.apply()` for more details.
            gd (CoefLike | None, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): Whether to check the vector. Defaults to True.

        Raises:
            RuntimeError: If gd is `None` and no default gd exists.

        Returns:
            TensorLike: New adjusted right-hand-size vector.
        """
        A = matrix
        f = self.check_vector(vector) if check else vector
        gd = self.gd if gd is None else gd

        if gd is None:
            raise RuntimeError("The boundary condition is None.")
        bd_idx = self.boundary_dof_index

        if uh is None:
            uh = bm.zeros_like(f)
        uh, isDDof = self.space.boundary_interpolate(gd, uh, self.threshold)

        if uh.ndim == 1:
            f = f - A.matmul(uh[:])
            f = bm.set_at(f, bd_idx, uh[bd_idx])

        elif uh.ndim == 2:
            if self.left:
                uh = bm.swapaxes(uh, 0, 1)
                f = bm.swapaxes(f, 0, 1)
            f = f - A.matmul(uh)
            f = bm.set_at(f, bd_idx, uh[bd_idx])
            if self.left:
                f = bm.swapaxes(f, 0, 1)
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

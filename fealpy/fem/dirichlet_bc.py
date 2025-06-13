
from typing import Optional, Tuple, Callable, Union, TypeVar

from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import SparseTensor, COOTensor, CSRTensor, spdiags
from ..functionspace.space import FunctionSpace

CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]
_ST = TypeVar('_ST', bound=SparseTensor)


class DirichletBC():
    """Dirichlet boundary condition."""
    def __init__(self, space: Tuple[FunctionSpace, ...],
                 gd: Optional[Tuple[CoefLike,...]]=None,
                 *, threshold: Optional[Tuple[CoefLike,...]]=None,
                 method = None):
        self.space = space
        self.gd = gd
        self.threshold = threshold
        self.bctype = 'Dirichlet'
        self.method = method

        if isinstance(space, tuple):
            self.gdof = bm.array([i.number_of_global_dofs() for i in space])
            if isinstance(threshold, tuple):
                self.is_boundary_dof = []
                for i in range(len(threshold)):
                    self.is_boundary_dof.append(space[i].is_boundary_dof(threshold[i], method=method))
                self.is_boundary_dof = bm.concatenate(self.is_boundary_dof)
            else:
                if threshold is None:
                    self.threshold = [None for i in range(len(space))]
                    self.is_boundary_dof = [i.is_boundary_dof() for i in space]
                    self.is_boundary_dof = bm.concatenate(self.is_boundary_dof)
                else:
                    index = bm.concatenate((bm.array([0]),bm.cumsum(self.gdof, axis=0)))
                    self.threshold = [threshold[i:i+1] for i in range(len(index)-1)]
                    self.is_boundary_dof = threshold 
            self.boundary_dof_index = bm.nonzero(self.is_boundary_dof)[0]
            self.gdof = bm.sum(self.gdof)
        else:
            self.gdof = space.number_of_global_dofs()
            if isinstance(threshold, TensorLike):
                self.is_boundary_dof = threshold
            else:
                self.is_boundary_dof = space.is_boundary_dof(threshold=threshold, method=method)

            self.boundary_dof_index = bm.nonzero(self.is_boundary_dof)[0]

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
        if vector.shape[0] != self.gdof:
            raise ValueError('The vector size must match the gdof of the space.')
        return vector

    def apply(self, A: SparseTensor, f: TensorLike, uh: Optional[TensorLike]=None,
              gd: Optional[CoefLike]=None, *,
              check=True) -> Tuple[TensorLike, TensorLike]:
        """Apply Dirichlet boundary conditions.

        Parameters:
            A (SparseTensor): Left-hand-size sparse matrix.
            f (Tensor): Right-hand-size vector.
            uh (Tensor | None, optional): The solution uh Tensor. Boundary interpolation\
                will be done on `uh` if given, which is an **in-place** operation.\
                Defaults to None.
            gd (CoefLike | None, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): _description_. Defaults to True.

        Returns:
            out (SparseTensor, Tensor): New adjusted `A` and `f`.
        """
        f = self.apply_vector(f, A, uh, gd, check=check)
        A = self.apply_matrix(A, check=check)
        return A, f

    def apply_matrix(self, matrix: _ST, *, check=True) -> _ST:
        """Apply Dirichlet boundary condition to left-hand-size matrix only.

        Parameters:
            matrix (SparseTensor): The original left-hand-size sparse matrix\
                of the linear system.
            check (bool, optional): Whether to check the matrix. Defaults to True.

        Returns:
            SparseTensor: New adjusted left-hand-size matrix.
        """
        A = self.check_matrix(matrix) if check else matrix
        isDDof = self.is_boundary_dof
        kwargs = A.values_context()
        bdIdx = bm.zeros(A.shape[0], **kwargs)
        # bdIdx[isDDof.reshape(-1)] = 1
        bdIdx = bm.set_at(bdIdx, isDDof.reshape(-1), 1)
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A

    def apply_vector(self, vector: TensorLike, matrix: SparseTensor,
                     uh: Optional[TensorLike]=None,
                     gd: Optional[CoefLike]=None, *, check=True) -> TensorLike:
        """Apply Dirichlet boundary contition to right-hand-size vector only.

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
        A = self.check_matrix(matrix) if check else matrix
        f = self.check_vector(vector) if check else vector
        gd = self.gd if gd is None else gd
        

        if gd is None:
            raise RuntimeError("The boundary condition is None.")
        
        if isinstance(self.space, tuple):
            if isinstance(gd, tuple):
                assert len(gd) == len(self.space)
                assert uh is None
                uh = []
                for i in range(len(gd)):
                    suh, sidDDdof = self.space[i].boundary_interpolate(gd=gd[i],
                                                                    threshold=self.threshold[i], method=self.method)
                    uh.append(suh[:])
                uh = bm.concatenate(uh)
            else:
                assert len(gd) == self.gdof
                uh = gd
        else:
            if uh is None:
                uh = bm.zeros_like(f)
            uh, _ = self.space.boundary_interpolate(gd=gd,uh=uh,
                                                threshold=self.threshold, method=self.method)
        bd_idx = self.boundary_dof_index
        f = f - A.matmul(uh[:])
        f = bm.set_at(f, bd_idx, uh[bd_idx])
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


# backup
def apply_csr_matrix(A: CSRTensor, isDDof: TensorLike):
    isIDof = bm.logical_not(isDDof)
    crow = A.crow
    col = A.col
    indices_context = bm.context(col)
    ZERO = bm.array([0], **indices_context)

    nnz_per_row = crow[1:] - crow[:-1]
    remain_flag = bm.repeat(isIDof, nnz_per_row) & isIDof[col] # 保留行列均为内部自由度的非零元素
    rm_cumsum = bm.concat([ZERO, bm.cumsum(remain_flag, axis=0)], axis=0) # 被保留的非零元素数量累积
    nnz_per_row = rm_cumsum[crow[1:]] - rm_cumsum[crow[:-1]] + isDDof # 计算每行的非零元素数量

    new_crow = bm.cumsum(bm.concat([ZERO, nnz_per_row], axis=0), axis=0)

    NNZ = new_crow[-1]
    non_diag = bm.ones((NNZ,), dtype=bm.bool, device=bm.get_device(isDDof)) # Field: non-zero elements
    loc_flag = bm.logical_and(new_crow[:-1] < NNZ, isDDof)
    non_diag = bm.set_at(non_diag, new_crow[:-1][loc_flag], False)

    new_col = bm.empty((NNZ,), **indices_context)
    new_col = bm.set_at(new_col, new_crow[:-1][loc_flag], isDDof)
    new_col = bm.set_at(new_col, non_diag, col[remain_flag])

    new_values = bm.empty((NNZ,), **A.values_context())
    new_values = bm.set_at(new_values, new_crow[:-1][loc_flag], 1.)
    new_values = bm.set_at(new_values, non_diag, A.values[remain_flag])

    return CSRTensor(new_crow, new_col, new_values, A.sparse_shape)

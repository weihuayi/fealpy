from scipy.sparse import spdiags

from fealpy.backend import backend_manager as bm
from fealpy.sparse import SparseTensor, COOTensor, CSRTensor

class VectorDirichletBC:
    def __init__(self, space, gd, threshold, direction = None):
        self.space = space
        self.gd = gd
        self.threshold = threshold
        self.direction = direction

    def set_boundary_dof(self):
        threshold = self.threshold  # Boundary detection function
        direction = self.direction   # Direction for applying boundary condition
        space = self.space
        mesh = space.mesh
        ipoints = mesh.interpolation_points(p=space.p)  # Interpolation points
        is_bd_dof = threshold(ipoints)  # Boolean mask for boundary DOFs
        GD = mesh.geo_dimension()  # Geometric dimension (2D or 3D)

        # Prepare an index array with shape (GD, npoints)
        index = bm.zeros((GD, ipoints.shape[0]), dtype=bool)
        #index = bm.zeros((ipoints.shape[0], GD), dtype=bool)

        # Map direction to axis: 'x' -> 0, 'y' -> 1, 'z' -> 2 (for GD = 3)
        direction_map = {'x': 0, 'y': 1, 'z': 2}

        if direction is None:
            bm.set_at(index, slice(None), is_bd_dof)  # Apply to all directions
            #bm.set_at(index, slice(None), is_bd_dof[..., None])  # Apply to all directions
        else:
            idx = direction_map.get(direction)
            if idx is not None and idx < GD:
                bm.set_at(index, idx, is_bd_dof)  # Apply only to the specified direction
                #bm.set_at(index, (..., idx), is_bd_dof)
            else:
                raise ValueError(f"Invalid direction '{direction}' for GD={GD}. Use 'x', 'y', 'z', or None.")
    
        # Flatten the index to return as a 1D array
        return index.ravel()

    def apply_value(self, u):
        index = self.set_boundary_dof()
        bm.set_at(u, index, self.gd) 
        return u, index   

    def apply(self, A, f, u=None):
        """
        Apply Dirichlet boundary condition to the linear system.

        Parameters
        ----------
        A : SparseTensor
            The coefficient matrix.
        f : TensorLike
            The right-hand-side vector.

        Returns
        -------
        A : SparseTensor
            The new coefficient matrix.
        f : TensorLike
            The new right-hand-side vector.
        """
        isDDof = self.set_boundary_dof()
        boundary_dof_index = bm.nonzero(isDDof)[0]

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
            isIDof = bm.logical_not(isDDof)
            crow = A.crow()
            col = A.col()
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
            new_col = bm.set_at(new_col, new_crow[:-1][loc_flag], boundary_dof_index)
            new_col = bm.set_at(new_col, non_diag, col[remain_flag])

            new_values = bm.empty((NNZ,), **kwargs)
            new_values = bm.set_at(new_values, new_crow[:-1][loc_flag], 1.)
            new_values = bm.set_at(new_values, non_diag, A.values()[remain_flag])

            A = CSRTensor(new_crow, new_col, new_values)

        bm.set_at(f, isDDof, self.gd) 
        return A, f



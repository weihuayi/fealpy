from scipy.sparse import spdiags

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.sparse import SparseTensor, COOTensor, CSRTensor

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

        # Map direction to axis: 'x' -> 0, 'y' -> 1, 'z' -> 2 (for GD = 3)
        direction_map = {'x': 0, 'y': 1, 'z': 2}

        if direction is None:
            bm.set_at(index, slice(None), is_bd_dof)  # Apply to all directions
        else:
            idx = direction_map.get(direction)
            if idx is not None and idx < GD:
                bm.set_at(index, idx, is_bd_dof)  # Apply only to the specified direction
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

        bm.set_at(f, isDDof, self.gd) 
        return A, f



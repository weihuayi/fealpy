import math

from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, SparseTensor
from ..mesh import UniformMesh

from .operator_base import OpteratorBase, assemblymethod

class LaplaceOperator(OpteratorBase):
    """
    LaplaceOperator constructs and assembles the discrete minus Laplace operator
    on a structured mesh for finite difference approximation.
    """
    def __init__(self, mesh: UniformMesh, method: Optional[str]=None):
        """
        Initialize the Laplace operator with a given structured mesh.

        Parameters:
            mesh (UniformMesh): Structured mesh object providing grid metadata.
        """
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.mesh = mesh  # Store the mesh for later assembly

    def assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Laplace operator.

        Returns:
            csr_matrix: Sparse matrix of size (NN, NN), where NN is number of nodes.
        """
        mesh = self.mesh
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        # spacing of the mesh in each dimension
        h = mesh.h
        # coefficient c = 1/h^2 per dimension
        c = 1.0 / (h ** 2)

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Create diagonal entries with sum of c over dimensions times 2
        diag_value = bm.full((NN,), 2 * c.sum().item(), dtype=ftype)
        I = K.ravel()  # Row indices for diagonal entries
        A = csr_matrix((diag_value, (I, I)), shape=(NN, NN))

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value = bm.full((NN - n_shift,), -c[i].item(), dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].ravel()
            J = K[s2].ravel()
            # Add entries for coupling in both directions
            A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix((off_value, (J, I)), shape=(NN, NN))

        return A

    @assemblymethod('fast')
    def fast_assembly(self) -> SparseTensor:
        pass

    def __matmul__(self, u: TensorLike) -> TensorLike:
        """
        Apply the assembled Laplace operator to a vector or tensor u via 
        matrix-vector product.

        Parameters:
            u (TensorLike): Input field values on mesh nodes.

        Returns:
            TensorLike: Resulting field after applying Laplace operator.
        """
        # TODO: Implement the matrix-vector multiplication using backend routines
        pass

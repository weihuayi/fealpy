import math
from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, SparseTensor
from ..mesh import UniformMesh
from .operator_base import assemblymethod, OpteratorBase

class WaveOperator(OpteratorBase):
    """
    WaveOperator constructs and assembles the discrete Wave operator 
    on a structured mesh for finite difference approximation.
    """
    def __init__(self, mesh: UniformMesh, method: Optional[str]=None):
        """
        Initialize the Wave operator with a given structured mesh.

        Parameters:
            mesh (UniformMesh): Structured mesh object providing grid metadata.
        """
        method = 'assembly' if (method is None) else method                     
        super().__init__(method=method)

        self.mesh = mesh  # Store the mesh for later assembly

    def assembly(self, tau: float, a: float = 1.0) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Wave operator.

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
        # The mesh ratio is given by a*tau/h
        r = a * tau / h
        
        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        diag_value = bm.full(NN, 2 - 2*(r**2).sum(), dtype=ftype)
        I = K.flat  # Row indices for diagonal entries
        J = K.flat  # Column indices for diagonal entries
        A = csr_matrix((diag_value, (I, J)), shape=(NN, NN))

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value = bm.full(NN - n_shift, (r[i])**2, dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix((off_value, (J, I)), shape=(NN, NN))

        return A
    
    @assemblymethod('implicit')
    def implicit_assembly(self, tau: float, a: float = 1.0, theta: float = 0.25) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Wave operator.

    Returns:
        csr_matrix: Three sparse matrices of size (NN, NN), where NN is the number of nodes.
        """
        mesh = self.mesh
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        # spacing of the mesh in each dimension
        h = mesh.h
        # The mesh ratio is given by a*tau/h
        r = a * tau / h

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        diag_value0 = bm.full(NN, 1 + 2 * theta * (r**2).sum(), dtype=ftype)
        diag_value1 = bm.full(NN, 2 - 2 * (1 - 2 * theta) * (r**2).sum(), dtype=ftype)
        diag_value2 = bm.full(NN, -1 - 2 * theta * (r**2).sum(), dtype=ftype)
        I = K.flat  # Row indices for diagonal entries
        J = K.flat  # Column indices for diagonal entries
        A0 = csr_matrix((diag_value0, (I, J)), shape=(NN, NN))
        A1 = csr_matrix((diag_value1, (I, J)), shape=(NN, NN))
        A2 = csr_matrix((diag_value2, (I, J)), shape=(NN, NN))

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value0 = bm.full(NN - n_shift, -theta * (r[i])**2, dtype=ftype)
            off_value1 = bm.full(NN - n_shift, (1 - 2 * theta) * (r[i])**2, dtype=ftype)
            off_value2 = bm.full(NN - n_shift, theta * (r[i])**2, dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            A0 += csr_matrix((off_value0, (I, J)), shape=(NN, NN))
            A0 += csr_matrix((off_value0, (J, I)), shape=(NN, NN))

            A1 += csr_matrix((off_value1, (I, J)), shape=(NN, NN))
            A1 += csr_matrix((off_value1, (J, I)), shape=(NN, NN))

            A2 += csr_matrix((off_value2, (I, J)), shape=(NN, NN))
            A2 += csr_matrix((off_value2, (J, I)), shape=(NN, NN))
        
        return A0, (A1, A2)

    def __matmul__(self, u: TensorLike) -> TensorLike:
        """
        Apply the assembled Wave operator to a vector or tensor u via 
        matrix-vector product.

        Parameters:
            u (TensorLike): Input field values on mesh nodes.

        Returns:
            TensorLike: Resulting field after applying Wave operator.
        """
        # TODO: Implement the matrix-vector multiplication using backend routines
        pass

import math

from typing import Literal
from .operator_base import OpteratorBase,assemblymethod
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, SparseTensor
from ..mesh import UniformMesh


class ParabolicOperator(OpteratorBase):
    """
    ParabolicOperator constructs and assembles the discrete Parabolic operator
    on a structured mesh for finite difference approximation.
    """
    def __init__(self, mesh: UniformMesh,
                 tau: float,
                 coef: float = None,
                 method: Literal['backward', 'forward', 'cn', None] = None):
        """
        Initialize the Parabolic operator with a given structured mesh.

        Parameters:
            mesh (UniformMesh): Structured mesh object providing grid metadata.
            coef (float): Coefficient for the parabolic operator.
            method (str): Method for time-stepping ('backward', 'forward', 'cn') default to 'forward' or None.
        """
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.mesh = mesh  # Store the mesh for later assembly
        self.tau = tau  # Time step size for the parabolic operator
        self.coef = coef  # Coefficient for the parabolic operator
        

    def assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Parabolic operator.

        Returns:
            csr_matrix: Sparse matrix of size (NN, NN), where NN is number of nodes.
        """
        mesh = self.mesh
        coef = self.coef
        tau = self.tau
        if coef is None:
            coef = 1.0
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        # spacing of the mesh in each dimension
        h = mesh.h
        # coefficient c = coef * tau / (h ^ 2) per dimension
        c =  tau / (h ** 2)
        if c.sum() > 0.5:
            raise ValueError(f"The r: {c.sum()} should be smaller than 0.5")
        c *= coef
        
        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Create diagonal entries with sum of c over dimensions times 2
        A_diag_value = bm.full(NN, 1, dtype=ftype)
        B_diag_value = bm.full(NN,1-2 * c.sum(), dtype=ftype)
        I = K.flat  # Row indices for diagonal entries
        J = K.flat  # Column indices for diagonal entries
        A = csr_matrix((A_diag_value, (I, J)), shape=(NN, NN))
        B = csr_matrix((B_diag_value, (I, J)), shape=(NN, NN))

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value = bm.full(NN - n_shift, c[i], dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            B += csr_matrix((off_value, (I, J)), shape=(NN, NN))
            B += csr_matrix((off_value, (J, I)), shape=(NN, NN))

        return A, (B,tau)
    
    @assemblymethod(call_name='backward')
    def backward_assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Parabolic operator.

        Returns:
            csr_matrix: Sparse matrix of size (NN, NN), where NN is number of nodes.
        """
        mesh = self.mesh
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh
        tau = self.tau # Time step size

        # spacing of the mesh in each dimension
        h = mesh.h
        # coefficient c = 1/h^2 per dimension
        c = tau/ (h ** 2)

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Create diagonal entries with sum of c over dimensions times 2
        A_diag_value = bm.full(NN, 1+ 2 * c.sum(), dtype=ftype)
        B_diag_value = bm.full(NN, 1, dtype=ftype)
        I = K.flat  # Row indices for diagonal entries
        J = K.flat  # Column indices for diagonal entries
        A = csr_matrix((A_diag_value, (I, J)), shape=(NN, NN))
        B = csr_matrix((B_diag_value, (I, J)), shape=(NN, NN))
        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value = bm.full(NN - n_shift, -c[i], dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix((off_value, (J, I)), shape=(NN, NN))

        return A,(B,tau)
    
    @assemblymethod(call_name='cn')
    def cn_assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Crank-Nicolson Parabolic operator.

        Parameters:
            tau (float): Time step size for the Crank-Nicolson method.

        Returns:
            A(csr_matrix): Sparse matrix of size (NN, NN), where NN is number of nodes.
            B(csr_matrix): Sparse matrix of size (NN, NN), where NN is number of nodes.
        """
        mesh = self.mesh
        tau = self.tau
        coef = self.coef
        ftype = mesh.ftype
        if coef is None:
            coef = 1.0
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        # spacing of the mesh in each dimension
        h = mesh.h
        # coefficient c = coef * tau / (h ^ 2) per dimension
        c = coef * tau / (h ** 2)

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Create diagonal entries with sum of c over dimensions times 2
        A_diag_value = bm.full(NN, 1 + c.sum(), dtype=ftype)
        B_diag_value = bm.full(NN, 1 - c.sum(), dtype=ftype)
        I = K.flat  # Row indices for diagonal entries
        J = K.flat  # Column indices for diagonal entries
        A = csr_matrix((A_diag_value, (I, J)), shape=(NN, NN))
        B = csr_matrix((B_diag_value, (I, J)), shape=(NN, NN))

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            A_off_value = bm.full(NN - n_shift, -c[i]/2, dtype=ftype)
            B_off_value = bm.full(NN - n_shift, c[i]/2, dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            A += csr_matrix((A_off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix((A_off_value, (J, I)), shape=(NN, NN))
            B += csr_matrix((B_off_value, (I, J)), shape=(NN, NN))
            B += csr_matrix((B_off_value, (J, I)), shape=(NN, NN))
        
        return A, (B,tau)

    def __matmul__(self, u: TensorLike) -> TensorLike:
        """
        Apply the assembled Parabolic operator to a vector or tensor u via 
        matrix-vector product.

        Parameters:
            u (TensorLike): Input field values on mesh nodes.

        Returns:
            TensorLike: Resulting field after applying Parabolic operator.
        """
        # TODO: Implement the matrix-vector multiplication using backend routines
        pass

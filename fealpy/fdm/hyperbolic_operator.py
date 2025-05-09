from typing import Literal
from .operator_base import OpteratorBase, assemblymethod
import math
from ..backend import backend_manager as bm
from ..sparse import csr_matrix, SparseTensor
from ..mesh import UniformMesh


class HyperbolicOperator(OpteratorBase):
    """
    HyperbolicOperator constructs and assembles the discrete hyperbolic operator
    on a structured mesh for finite difference approximation.
    """
    def __init__(self, mesh: UniformMesh, 
                 tau: float, 
                 a:float,
                 method: Literal['lax_friedrichs', 
                                 'central', 
                                 'explicity_upwind_viscous', None] = None):
        """
        Initialize the Hyperbolic operator with a given structured mesh.
        Parameters:
            mesh (UniformMesh): Structured mesh object providing grid metadata.
        """
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.mesh = mesh  # Store the mesh for later assembly
        self.tau = tau  # Time step size
        self.a = a  # Smoothing parameter
        
    def assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Hyperbolic operator.
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
        r = self.a*self.tau/h
        if r.sum() > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")
        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array
        # Create diagonal entries with sum of c over dimensions times 2
        if self.a > 0:
            diag_value = bm.full(NN, 1 - r.sum(), dtype=ftype)
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
                off_value = bm.full(NN - n_shift, r[i], dtype=ftype)
                # Create slice objects to select neighbor index arrays
                s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
                s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
                # Row indices for off-diagonal
                I = K[s1].flat
                J = K[s2].flat
                # Add entries for coupling in both directions
                if GD == 1:
                    A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
                else:
                    A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
                    #A += csr_matrix((off_value, (J, I)), shape=(NN, NN))
        else:
            diag_value = bm.full(NN, 1 + r.sum(), dtype=ftype)
            I = K.flat  # Row indices for diagonal entries
            J = K.flat  # Column indices for diagonal entries
            A = csr_matrix((diag_value, (I, J)), shape=(NN, NN))
            full_slice = (slice(None),) * GD
            # Off-diagonal contributions for each dimension
            for i in range(GD):
                # Number of nodes shifted along dimension i
                n_shift = math.prod(
                    count for dim_idx, count in enumerate(shape) if dim_idx != i
                )
                # Off-diagonal value for neighbor entries
                off_value = bm.full(NN - n_shift, -r[i], dtype=ftype)
                # Create slice objects to select neighbor index arrays
                s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
                s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
                # Row indices for off-diagonal
                I = K[s1].flat
                J = K[s2].flat
                # Add entries for coupling in both directions
                if GD == 1:
                    A += csr_matrix((off_value, (J, I)), shape=(NN, NN))
                else:
                    #A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
                    A += csr_matrix((off_value, (J, I)), shape=(NN, NN))
        return A
        
    @assemblymethod(call_name='lax_friedrichs')
    def lax_friedrichs_assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Hyperbolic operator.
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
        r = self.a*self.tau/h
        if r.sum() > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")
        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array
        diag_value = bm.full(NN, 0, dtype=ftype)
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
            off_value0 = bm.full(NN - n_shift, 1/2 *(1 + r[i]), dtype=ftype)
            off_value1 = bm.full(NN - n_shift, 1/2 *(1 - r[i]), dtype=ftype)            
            off_value2 = bm.full(NN - n_shift, 1/4 + 1/2*r[i], dtype=ftype)
            off_value3 = bm.full(NN - n_shift, 1/4 - 1/2*r[i], dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            if GD == 1:
                A += csr_matrix((off_value0, (I, J)), shape=(NN, NN))
                A += csr_matrix((off_value1, (J, I)), shape=(NN, NN))
            else:
                A += csr_matrix((off_value2, (I, J)), shape=(NN, NN))
                A += csr_matrix((off_value3, (J, I)), shape=(NN, NN))
        return A
    
    @assemblymethod(call_name='central')
    def central_assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Hyperbolic operator.
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
        r = self.a*self.tau/h
        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array
        diag_value = bm.full(NN, 1-abs(r.sum()), dtype=ftype)
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
            off_value0 = bm.full(NN - n_shift, r[i]/2, dtype=ftype)
            off_value1 = bm.full(NN - n_shift, -r[i]/2, dtype=ftype)            
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
       
            A += csr_matrix((off_value0, (I, J)), shape=(NN, NN))
            A += csr_matrix((off_value1, (J, I)), shape=(NN, NN))

        return A
    
    @assemblymethod(call_name='explicity_upwind_viscous')
    def explicity_upwind_viscous_assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Hyperbolic operator.
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
        r = self.a*self.tau/h
        if r.sum() > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")
        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array
        diag_value = bm.full(NN, 1 - r.sum(), dtype=ftype)
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
            off_value0 = bm.full(NN - n_shift, r[i], dtype=ftype)
            off_value1 = bm.full(NN - n_shift, 0, dtype=ftype)            
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (
                slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            if GD == 1:
                A += csr_matrix((off_value0, (I, J)), shape=(NN, NN))
                A += csr_matrix((off_value1, (J, I)), shape=(NN, NN))
            else:
                A += csr_matrix((off_value0, (I, J)), shape=(NN, NN))
                A += csr_matrix((off_value1, (J, I)), shape=(NN, NN))
                
        return A






    

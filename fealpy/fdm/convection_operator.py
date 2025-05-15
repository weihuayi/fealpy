import math

from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, spdiags, SparseTensor
from ..mesh import UniformMesh

from .operator_base import OpteratorBase, assemblymethod

class ConvectionOperator(OpteratorBase):
    """
    """
    def __init__(self, mesh: UniformMesh, 
                 convection_coef,
                 method: Optional[str]=None):
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.mesh = mesh  # Store the mesh for later assembly
        self.convection_coef = convection_coef

    def assembly(self) -> SparseTensor:
        """
        """
        mesh = self.mesh
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        node = self.mesh.entity('node')
        b = self.convection_coef(node) # shape == (GD, )

        # spacing of the mesh in each dimension
        h = mesh.h
        c = b / h / 2.0 

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        val = bm.zeros(NN, dtype=ftype)
        A = spdiags(val, 0, NN, NN, format='csr') 
        
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
            A += csr_matrix((-off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix(( off_value, (J, I)), shape=(NN, NN))
        return A

    @assemblymethod('wind')
    def assembly_const_wind(self) -> SparseTensor:
        pass

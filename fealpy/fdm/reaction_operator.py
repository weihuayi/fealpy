import math

from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, spdiags, SparseTensor
from ..mesh import UniformMesh

from .operator_base import OpteratorBase, assemblymethod

class ReactionOperator(OpteratorBase):
    """
    """
    def __init__(self, mesh: UniformMesh, reaction_coef,
                 method: Optional[str]=None):
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.mesh = mesh  # Store the mesh for later assembly
        self.reaction_coef = reaction_coef

    def assembly(self) -> SparseTensor:
        """
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        c = self.reaction_coef(mesh.entity('node'))
        val = bm.full(NN, c, dtype=mesh.ftype)
        D = spdiags(val, 0, NN, NN, format='csr')
        return D

import math
from typing import Optional
import inspect
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, spdiags, SparseTensor
from ..mesh import UniformMesh

from .operator_base import OpteratorBase, assemblymethod

class ReactionOperator(OpteratorBase):
    """
    Discrete approximation of the reaction operator:

        R(x) · u(x)

    on structured (uniform Cartesian) meshes using finite difference discretization.

    This term corresponds to spatially varying (or constant) pointwise multiplication
    in many PDEs, such as:
        - reaction-diffusion equations
        - source terms in parabolic/elliptic equations
        - penalization or damping terms

    -------------------------------------------------------------------------------
    Method extensibility:
        Alternative implementations may be added via the `assemblymethod` decorator.

    -------------------------------------------------------------------------------
    Attributes:
        mesh           : UniformMesh object
        reaction_coef  : Callable or constant defining R(x) at each node
    """

    def __init__(self,
                 mesh: UniformMesh,
                 reaction_coef,
                 method: Optional[str] = None):
        """
        Initialize the reaction operator.

        Parameters:
            mesh           : The structured mesh where the operator is defined.
            reaction_coef  : Callable or constant tensor defining the reaction term R(x).
                             Should return a 1D tensor of shape (NN,) evaluated at all nodes.
            method         : Optional string key for choosing implementation method.
        """
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.mesh = mesh
        self.reaction_coef = reaction_coef

    def assembly(self) -> SparseTensor:
        """
        Assemble the diagonal matrix for the reaction operator.

        The operator acts pointwise on the solution vector as:
            R(x) * u(x)  →  diag(R) · u

        Returns:
            SparseTensor: CSR-format sparse diagonal matrix of shape (NN, NN),
                          where NN is the number of mesh nodes.
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        # Evaluate reaction coefficient at each node
        f = self.reaction_coef
        l = len(inspect.signature(f).parameters)
        if callable(self.reaction_coef) :
            if l == 2:
                c = self.reaction_coef(mesh.entity('node')) # shape:(NN,)
                data = c    
            else:
                c = self.reaction_coef() # shape:(1,)
                data = bm.full(NN, c)
                
        else:
            c = self.reaction_coef 
            data = bm.full(NN, c)
            
        D = spdiags(data, 0, NN, NN, format='csr')
        return D


from typing import Optional, Tuple, Callable, Union

from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import COOTensor
from ..functionspace.space import FunctionSpace

from .form import Form

CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]

class DirichletBCOperator():
    """Dirichlet boundary condition operator."""
    def __init__(self, form: Form,
                 gd: Optional[CoefLike]=None,
                 *, threshold: Optional[Callable]=None, 
                 isDDof=None,
                 left:bool=True):
        self.form = form
        self.gd = gd
        if isDDof is None:
            isDDof = form._spaces[0].is_boundary_dof(threshold=threshold) # on the same device as space
            self.is_boundary_dof = isDDof
        else :
            self.is_boundary_dof = isDDof
        self.boundary_dof_index = bm.nonzero(isDDof)[0]
        self.shape = form.shape 

    def init_solution(self):
        """
        Generate the init solution with correct Dirichlet boundary
        condition.

        Returns:
            u (TensorLike): the init solution.
        TODO:
            1. deal with device
        """
        uh = bm.zeros(self.shape[1], dtype=self.form._spaces[0].ftype)
        self.form._spaces[0].boundary_interpolate(self.gd, uh,
                threshold=self.is_boundary_dof)
        return uh

    def apply(self, F, uh):
        F = F - self.form @ uh 
        F = bm.set_at(F, self.is_boundary_dof, uh[self.is_boundary_dof])
        return F

    def __matmul__(self, u: TensorLike):
        """Apply the dirichlet boundary condition on the matrix-vetor multiply.

        Parameters:
            u (TensorLike): the input vector.

        Returns:
            v (TensorLike): the result of matrix-vector multiply.

        TODO:
            1. support for v.shape[0] != u.shape[0]
        """
        v = bm.copy(u) 
        val = v[self.is_boundary_dof]
        bm.set_at(v, self.is_boundary_dof, 0.0)
        v = self.form @ v 
        bm.set_at(v, self.is_boundary_dof, val) 
        return v

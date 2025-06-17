
from typing import Tuple, Callable, Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike as Tensor
from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarNeumannBCIntegrator
)
from fealpy.sparse import COOTensor
from fealpy.solver import cg, spsolve


class EITDataGenerator():
    """Generate boundary voltage and current data for EIT.
    """
    def __init__(self, mesh: Mesh, p: int = 1, q: Optional[int] = None) -> None:
        """Create a new EIT data generator.

        Args:
            mesh (Mesh): _description_
            p (int, optional): Order of the Lagrange finite element space. Defaults to 1.
            q (int | None, optional): Order of the quadrature, use `q = p + 2` if None.
                Defaults to None.
        """
        q = p + 2 if q is None else q

        # setup function space
        kwargs = dict(dtype=mesh.itype, device=mesh.device)
        space = LagrangeFESpace(mesh, p=p) # FE space
        self.space = space

        # fetch boundary nodes to output gd and gn
        bd_node_index = mesh.boundary_node_index()
        bd_node = mesh.entity('node', index=bd_node_index) # (Q, C, 2)
        self.bd_node = bd_node
        self._bd_node_index = bd_node_index

        # initialize integrators
        self._bsi = ScalarNeumannBCIntegrator(None, q=q)
        self._di = ScalarDiffusionIntegrator(None, q=q)

        # prepare for the unique condition in the neumann case
        self.gdof = space.number_of_global_dofs()
        lform_c = LinearForm(space)
        lform_c.add_integrator(ScalarNeumannBCIntegrator(1.))
        self.cdata = lform_c.assembly()
        cdata_row = bm.arange(self.gdof, **kwargs)
        cdata_col = bm.full((self.gdof,), self.gdof, **kwargs)
        self.cdata_indices = bm.stack([cdata_row, cdata_col], axis=0)

    def set_levelset(self, sigma_vals: Tuple[float, float],
                     levelset: Callable[[Tensor], Tensor]) -> None:
        """Set inclusion distribution.

        Args:
            sigma_vals (Tuple[float, float]): Sigma value of inclusion and background.
            levelset (Callable): level-set function indicating inclusion and background.
        """
        def _coef_func(p: Tensor):
            inclusion = levelset(p) < 0. # a bool tensor on quadrature points.
            sigma = bm.empty(p.shape[:2], **bm.context(p)) # (Q, C)
            sigma = bm.set_at(sigma, inclusion, sigma_vals[0])
            sigma = bm.set_at(sigma, ~inclusion, sigma_vals[1])
            return sigma
        _coef_func.coordtype = getattr(levelset, 'coordtype', 'cartesian')

        space = self.space

        bform = BilinearForm(space)
        self._di.coef = _coef_func
        self._di.clear() # clear the cached result as the coef has changed
        bform.add_integrator(self._di)
        self._A = bform.assembly(format='coo')

        cdata_indices = self.cdata_indices
        cdataT_indices = bm.flip(cdata_indices, axis=0)
        A_n_indices = bm.concat([self._A.indices, cdata_indices, cdataT_indices], axis=1)
        A_n_values = bm.concat([self._A.values, self.cdata, self.cdata], axis=-1)
        A_n = COOTensor(A_n_indices, A_n_values, spshape=(self.gdof+1, self.gdof+1))
        self.A_n = A_n.tocsr()

    def set_boundary(self, gn_source: Union[Callable[[Tensor], Tensor], Tensor],
                     batch_size: int=0, *, zero_integral=False) -> Tensor:
        """Set boundary current density.

        Args:
            gn_source (Callable[[Tensor], Tensor] | Tensor): The current density\
                function or Tensor on the boundary.
            batch_size (int, optional): The batch size of the boundary current\
                density function. Defaults to 0.
            zero_integral (bool, optional): Whether zero the integral of the\
                current density on the boundary. Defaults to False.

        Returns:
            Tensor: current value on boundary nodes, shaped (Boundary nodes, )\
                or (Batch, Boundary nodes).
        """
        # NOTE: current values on boundary nodes are needed instead of
        # boundary faces, because we measure the current by the electric node
        # in the real world.
        # NOTE: The value measured on the node is actually 'current', not the
        # 'current density'. We assume the the current measured by the electric
        # node is the integral of the current density function.
        if callable(gn_source):
            lform = LinearForm(self.space, batch_size=batch_size)
            self._bsi.gn = gn_source
            self._bsi.batched = (batch_size > 0)
            self._bsi.clear()
            lform.add_integrator(self._bsi)
            b_ = lform.assembly()
            kwargs = bm.context(b_)
            current = b_[..., self._bd_node_index]
        else:
            current = gn_source
            kwargs = bm.context(current)
            b_ = bm.zeros((gn_source.shape[0], self.gdof))
            b_ = bm.set_at(b_, (slice(None), self._bd_node_index), current)

        if zero_integral:
            b_ = b_ - bm.mean(b_, axis=0)

        if b_.ndim == 1:
            ZERO = bm.zeros((1,), **kwargs)
        else:
            ZERO = bm.zeros((b_.shape[0], 1), **kwargs)
        self.b_ = bm.concat([b_, ZERO], axis=-1)

        return current

    def run(self, return_full=False) -> Tensor:
        """Generate voltage on boundary nodes.

        Args:
            return_full (bool, optional): Whether return all dofs. Defaults to False.

        Returns:
            Tensor: gd Tensor, shaped (Boundary nodes, )\
                or (Batch, Boundary nodes).
        """
        # uh = cg(self.A_n, self.b_, batch_first=True, atol=1e-12, rtol=0.)
        uh = spsolve(self.A_n, self.b_.T, solver='scipy').T

        if return_full:
            return uh[:-1]

        # NOTE: interpolation points on nodes are arranged firstly,
        # therefore the value on the boundary nodes can be fetched like this:
        return uh[..., self._bd_node_index] # voltage

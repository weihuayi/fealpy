
from typing import Tuple, Callable, Union, Optional

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import torch
from torch import Tensor, cat

from fealpy.torch.mesh import Mesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarBoundarySourceIntegrator
)


class EITDataGenerator():
    """Generate boundary voltage and current data for EIT.
    """
    def __init__(self, mesh: Mesh, p: int=1, q: Optional[int]=None) -> None:
        """Create a new EIT data generator.

        Args:
            mesh (Mesh): _description_
            p (int, optional): Order of the Lagrange finite element space. Defaults to 1.
            q (int | None, optional): Order of the quadrature, use `q = p + 2` if None.\
            Defaults to None.
        """
        q = p + 2 if q is None else q

        # setup function space
        kwargs = dict(dtype=mesh.ftype, device=mesh.device)
        space = LagrangeFESpace(mesh, p=p) # FE space
        self.space = space

        # fetch boundary nodes to output gd and gn
        bd_node_index = mesh.ds.boundary_node_index()
        bd_node = mesh.entity('node', index=bd_node_index) # (Q, C, 2)
        self.bd_node = bd_node
        self._bd_node_index = bd_node_index

        # initialize integrators
        self._bsi = ScalarBoundarySourceIntegrator(None, q=q, zero_integral=True)
        self._di = ScalarDiffusionIntegrator(None, q=q)

        # prepare for the unique condition in the neumann case
        bd_dof = space.is_boundary_dof().nonzero().ravel()
        gdof = space.number_of_global_dofs()
        zeros = torch.zeros_like(bd_dof, **kwargs)
        lform_c = LinearForm(space)
        lform_c.add_integrator(ScalarBoundarySourceIntegrator(1.))
        cdata = lform_c.assembly(return_dense=False)
        self.c = torch.sparse_coo_tensor(
            torch.stack([cdata.indices()[0], zeros], dim=0),
            cdata.values(),
            size=(gdof, 1)
        )
        self.ct = torch.sparse_coo_tensor(
            torch.stack([zeros, cdata.indices()[0]], dim=0),
            cdata.values(),
            size=(1, gdof)
        )
        self.ZERO = torch.sparse_coo_tensor(
            torch.zeros((2, 1), dtype=mesh.ds.itype, device=mesh.device),
            torch.zeros((1,), **kwargs),
            size=(1, 1)
        )

    def set_levelset(self, sigma_vals: Tuple[float, float],
                     levelset: Callable[[Tensor], Tensor]) -> Tensor:
        """Set inclusion distribution.

        Args:
            sigma_vals (Tuple[float, float]): Sigma value of inclusion and background.
            levelset (Callable): _description_.

        Retunrs:
            Tensor: Label boolean Tensor with True for inclusion nodes and False\
                for background nodes, shaped (nodes, ).
        """
        def _coef_func(p: Tensor):
            inclusion = levelset(p) < 0.
            sigma = torch.empty(p.shape[:2], dtype=p.dtype, device=p.device) # (Q, C)
            sigma[inclusion] = sigma_vals[0]
            sigma[~inclusion] = sigma_vals[1]
            return sigma

        space = self.space

        bform = BilinearForm(space)
        self._di.set_coef(_coef_func)
        bform.add_integrator(self._di)
        self._A = bform.assembly()
        A_n = cat([cat([self._A, self.c], dim=1),
                   cat([self.ct, self.ZERO], dim=1)], dim=0).coalesce()
        self._A_n_np = csr_matrix(
            (A_n.values().cpu().numpy(), A_n.indices().cpu().numpy()),
            shape=A_n.shape
        )

        node = space.mesh.entity('node')
        return levelset(node) < 0.

    def set_boundary(self, gn_source: Union[Callable[[Tensor], Tensor], Tensor],
                     batch_size: int=0) -> Tensor:
        """Set boundary current density.

        Args:
            gn_source (Callable[[Tensor], Tensor] | Tensor): _description_

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
        lform = LinearForm(self.space, batch_size=batch_size)
        self._bsi.set_source(gn_source)
        self._bsi.batched = (batch_size > 0)
        lform.add_integrator(self._bsi)
        b_ = lform.assembly()
        current = b_[:, self._bd_node_index]
        self.unsqueezed = False

        if b_.ndim == 1:
            b_ = b_.unsqueeze(0)
            self.unsqueezed = True

        NUM = b_.size(0)
        ZERO = torch.zeros((NUM, 1), dtype=b_.dtype, device=b_.device)
        self.b_ = torch.cat([b_, ZERO], dim=-1).cpu().numpy()

        return current

    def run(self, return_full=False) -> Tensor:
        """Generate voltage on boundary nodes.

        Args:
            return_full (bool, optional): Whether return all dofs. Defaults to False.

        Returns:
            Tensor: gd Tensor on **CPU**, shaped (Boundary nodes, )\
                or (Batch, Boundary nodes).
        """
        uh = spsolve(self._A_n_np, self.b_.T).T
        uh = torch.from_numpy(uh) # cpu

        if self.unsqueezed:
            uh = uh.squeeze(0)

        if return_full:
            return uh

        # NOTE: interpolation points on nodes are arranged firstly,
        # therefore the value on the boundary nodes can be fetched like this:
        return uh[..., self._bd_node_index.cpu()] # voltage

# TODO: finish this
class EITDataPreprocessor():
    """Process the EIT data from voltage & current to the neumann boundary condition
    of the data feature."""
    def __init__(self, mesh: Mesh, p: int=1, q: Optional[int]=None):
        q = q or p + 2

        # setup function space
        space = LagrangeFESpace(mesh, p=p, q=q)
        self.space = space

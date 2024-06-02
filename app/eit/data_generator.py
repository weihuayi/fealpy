
from typing import Tuple, Callable, Union

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
    """Generate boundary voltage and current density data for EIT.
    """
    def __init__(self, mesh: Mesh) -> None:
        """Create a new EIT data generator.

        Args:
            mesh (Mesh): _description_
        """
        # setup function space
        kwargs = dict(dtype=mesh.ftype, device=mesh.device)
        space = LagrangeFESpace(mesh, p=1) # Linear FE space
        self.space = space

        # fetch boundary nodes to output gd and gn
        bd_node_index = mesh.ds.boundary_node_index()
        bd_node = mesh.entity('node', index=bd_node_index) # (Q, C, 2)
        self.bd_node = bd_node
        self._bd_node_index = bd_node_index

        # fetch integrators
        self._bsi = ScalarBoundarySourceIntegrator(None, zero_integral=True, batched=True)
        self._di = ScalarDiffusionIntegrator(None)

        # prepare for the unique condition in the neumann case
        gdof = space.number_of_global_dofs()
        zeros = torch.zeros((gdof, ), **kwargs)
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
            Tensor: Label Tensor.
        """
        def _coef_func(p: Tensor):
            inclusion = levelset(p) < 0.
            sigma = torch.empty(p.shape[:2], dtype=p.dtype, device=p.device) # (Q, C)
            sigma[inclusion] = sigma_vals[0]
            sigma[~inclusion] = sigma_vals[1]
            return sigma

        space = self.space

        bform = BilinearForm(space)
        self._di.coef = _coef_func
        self._di.clear(result_only=True)
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
                     batched=True) -> Tensor:
        """Set boundary current density.

        Args:
            gn_source (Union[Callable[[Tensor], Tensor], Tensor]): _description_
            batched (bool, optional): _description_. Defaults to True.

        Returns:
            Tensor: current density on boundary nodes.
        """
        # NOTE: current density values on boundary nodes are needed instead of
        # boundary faces, because we measure the current by the electric node
        # in the real world.
        gn = gn_source(self.bd_node)
        batch_size = gn.size(0) if batched else 0
        lform = LinearForm(self.space, batch_size=batch_size)
        self._bsi.f = gn_source
        self._bsi.batched = batched
        self._bsi.clear(result_only=True)
        lform.add_integrator(self._bsi)
        b_ = lform.assembly()
        self.unsqueezed = False

        if b_.ndim == 1:
            b_ = b_.unsqueeze(0)
            self.unsqueezed = True

        NUM = b_.size(0)
        ZERO = torch.zeros((NUM, 1), dtype=b_.dtype, device=b_.device)
        self.b_ = torch.cat([b_, ZERO], dim=1).cpu().numpy()

        return gn

    def run(self, return_full=False) -> Tensor:
        """Generate voltage on boundary nodes.

        Args:
            return_full (bool, optional): Whether return full uh. Defaults to False.

        Returns:
            Tensor: gd Tensor on **CPU**.
        """
        uh = spsolve(self._A_n_np, self.b_.T).T
        uh = torch.from_numpy(uh) # cpu

        if self.unsqueezed:
            uh = uh.squeeze(0)

        if return_full:
            return uh

        # NOTE: interpolation points on nodes are arranged firstly,
        # therefore the value on the boundary nodes can be fetched like this:
        return uh[..., self._bd_node_index]

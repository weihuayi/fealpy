
from typing import Tuple, Callable, Union, Optional

import torch
from torch import Tensor, cat
from torch.linalg import lu_factor, lu_solve

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
    def __init__(self, mesh: Mesh,
                 sigma_vals: Tuple[float, float],
                 levelset: Callable[[Tensor], Tensor]) -> None:
        """Create a new EIT data generator.

        Args:
            mesh (Mesh): _description_
            sigma_vals (Tuple[float, float]): Sigma value of inclusion and background.
            levelset (Callable): _description_.
        """
        space = LagrangeFESpace(mesh, p=1)
        ips = space.interpolation_points() # (Q, C, 2)

        def _neumann_func(p: Tensor):
            inclusion = levelset(p) < 0.
            sigma = torch.empty(p.shape[:2], dtype=p.dtype, device=p.device) # (Q, C)
            sigma[inclusion] = sigma_vals[0]
            sigma[~inclusion] = sigma_vals[1]
            return sigma

        self.space = space
        self.ips = ips
        self.levelset = levelset
        self.bform = BilinearForm(space)
        self.bform.add_integrator(ScalarDiffusionIntegrator(_neumann_func))
        self._A = self.bform.assembly()
        self._bd_dof = space.is_boundary_dof()

        c = self._bd_dof.to(mesh.ftype)[None, :]
        ZERO = torch.zeros((1,1), dtype=mesh.ftype, device=mesh.device)
        _A_n = cat([cat([self._A.to_dense(), c], dim=0),
                    cat([c.T, ZERO], dim=0)], dim=1)
        self._A_n_lu, self.pivots_n = lu_factor(_A_n)

    def run(self, gn_source: Union[Callable[[Tensor], Tensor], Tensor],
            batched=True) -> Tuple[Tensor, Tensor]:
        gn = gn_source(self.ips[self._bd_dof])
        batch_size = gn.size(0) if batched else 0
        lform = LinearForm(self.space, batch_size=batch_size)
        lform.add_integrator(
            ScalarBoundarySourceIntegrator(gn_source, zero_integral=True, batched=batched)
        )
        b_ = lform.assembly()
        unsqueezed = False

        if b_.ndim == 1:
            b_ = b_.unsqueeze(0)
            unsqueezed = True

        NUM = b_.size(0)
        ZERO = torch.zeros((NUM, 1), dtype=b_.dtype, device=b_.device)
        b_ = torch.cat([b_, ZERO], dim=1)
        uh = lu_solve(self._A_n_lu, self.pivots_n, b_, left=False)[:, :-1]

        if unsqueezed:
            uh = uh.squeeze(0)

        return uh[..., self._bd_dof], gn

    def label(self):
        mesh = self.space.mesh
        node = mesh.entity('node')
        return self.levelset(node) < 0.

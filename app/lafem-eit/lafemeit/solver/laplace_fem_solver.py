
from typing import Optional

import torch
from torch import Tensor, cat
from torch.linalg import lu_factor, lu_solve

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import (
    ScalarDiffusionIntegrator,
    ScalarNeumannBCIntegrator,
    DirichletBC,
)


class LaplaceFEMSolver():
    def __init__(self, mesh: TriangleMesh, p: int = 1, q: Optional[int] = None, *,
                 is_sampler_dof: Optional[Tensor] = None, reserve_mat=False) -> None:
        """_summary_

        Args:
            mesh (TriangleMesh): FEALPy Mesh object.
            p (int, optional): Order of the FE space. Defaults to 1.
            q (int | None, optional): ID of the quadrature formula, using `p+2` if None. Defaults to None.
            is_sampler_dof (Tensor | None, optional): A bool Tensor indicating sampler dofs.\
                Use all the boundary dofs if None. Defaults to None.
            reserve_mat (bool, optional): _description_. Defaults to False.
        """
        assert bm.backend_name == 'pytorch', "FEALPy should work with PyTorch backend."

        q = q or p + 2
        self.space = LagrangeFESpace(mesh, p=p)
        self.dtype = mesh.ftype
        self.device = mesh.device
        self.reserve_mat = reserve_mat

        is_Ddof = self.space.is_boundary_dof()

        if is_sampler_dof is not None:
            assert torch.all(is_Ddof[is_sampler_dof]), "All sampler dofs should be boundary dofs"
            self.is_sampler_dof = is_sampler_dof
        else:
            self.is_sampler_dof = is_Ddof

        self.sampler_dof_id = torch.nonzero(self.is_sampler_dof, as_tuple=True)[0]
        self.gdof = self.space.number_of_global_dofs()

        # Generate left-hand-size matrix
        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=q))
        self._A = bform.assembly(format='csr')

    def _init_dirichlet(self):
        self.dbc = DirichletBC(self.space, threshold=self.is_sampler_dof)
        A_d = self.dbc.apply_matrix(self._A).to_dense()
        if self.reserve_mat:
            self.A_d = A_d
        self.A_d_LU, self.pivots_d = lu_factor(A_d)

    def _init_neumann(self):
        lform_c = LinearForm(self.space)
        lform_c.add_integrator(
            ScalarNeumannBCIntegrator(1.)
        )
        c = lform_c.assembly()[None, :]
        A = self._A.to_dense()
        ZERO = torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        A_n = cat([
            cat([A, c.T], dim=1),
            cat([c, ZERO], dim=1)
        ], dim=0)
        if self.reserve_mat:
            self.A_n = A_n
        self.A_n_LU, self.pivots_n = lu_factor(A_n)

    def solve_from_potential(self, potential: Tensor, /) -> Tensor:
        """Solve from boundary potential to global dofs."""
        if not hasattr(self, 'A_d_LU'):
            self._init_dirichlet()

        batch_size = potential.shape[0]
        # NOTE: this can work if the initial f is zero.
        f = torch.zeros((batch_size, self.gdof), dtype=self.dtype, device=self.device)
        f[:, self.sampler_dof_id] = potential
        f = f - self._A.matmul(f.T).T
        f[:, self.sampler_dof_id] = potential

        self._latest_fd = f
        uh = lu_solve(self.A_d_LU, self.pivots_d, f, left=False)
        return uh

    def solve_from_current(self, current: Tensor, /, *, remove_offset=False) -> Tensor:
        """Solve from boundary current to global dofs."""
        if not hasattr(self, 'A_n_LU'):
            self._init_neumann()

        batch_size = current.shape[0]
        f = torch.zeros((batch_size, self.gdof + 1), dtype=self.dtype, device=self.device)
        f[:, self.sampler_dof_id] = current

        if remove_offset:
            f[:, self.sampler_dof_id] -= f[:, self.sampler_dof_id].mean(dim=-1, keepdim=True)

        self._latest_fn = f
        uh = lu_solve(self.A_n_LU, self.pivots_n, f, left=False)
        return uh[..., :-1]

    def boundary_value(self, uh: Tensor, /) -> Tensor:
        """From global dofs to boundary potential."""
        return uh[..., self.sampler_dof_id]

    def normal_derivative(self, uh: Tensor, /) -> Tensor:
        """From global dofs to boundary current."""
        return self._A.matmul(uh.T).T[..., self.sampler_dof_id]

    def value_on_nodes(self, uh: Tensor, /) -> Tensor:
        """Find values on mesh nodes."""
        self.NUM_NODES = self.space.mesh.number_of_nodes()
        return uh[..., :self.NUM_NODES]

    def residual_fd(self, uh: Tensor, /) -> Tensor:
        if not hasattr(self, 'A_d'):
            raise RuntimeError("A_d is not initialized. Please call `init_gd` first "
                               "and make sure reserve_mat is set to True")
        diff = uh @ self.A_d - self._latest_fd
        return diff.norm(dim=-1).mean()

    def residual_fn(self, uh: Tensor, /) -> Tensor:
        if not hasattr(self, 'A_n'):
            raise RuntimeError("A_n is not initialized. Please call `init_gn` first "
                               "and make sure reserve_mat is set to True")
        ZERO = torch.zeros((uh.shape[0], 1), dtype=self.dtype, device=self.device)
        uh = torch.cat([uh, ZERO], dim=-1)
        diff = uh @ self.A_n - self._latest_fn
        return diff.norm(dim=-1).mean()

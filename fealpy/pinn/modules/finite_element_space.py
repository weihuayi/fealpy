
from typing import Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module

from ...functionspace import LagrangeFESpace


def shape_function(mesh, bc: Tensor, p: int=1):
    """
    @brief
    """
    TD = bc.shape[-1] - 1
    multiIndex = mesh.multi_index_matrix(p)
    c = np.arange(1, p+1, dtype=np.int_)
    P = 1.0/np.multiply.accumulate(c)
    t = torch.arange(0, p, dtype=torch.float32)
    shape = bc.shape[:-1]+(p+1, TD+1)

    A = torch.ones(shape, dtype=torch.float32)
    A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
    torch.cumprod(A, dim=-2, out=A)
    A[..., 1:, :] *= torch.from_numpy(P).reshape(-1, 1)
    idx = torch.arange(TD+1)
    phi = torch.prod(A[..., multiIndex, idx], dim=-1)
    return phi


class LagrangeFESpaceLayer(LagrangeFESpace, Module):
    """
    @brief The torch version of Lagrange Finite Element Space.
    """
    def basis_tensor(self, bc: Tensor, index=np.s_[:]):
        phi = shape_function(self.mesh, bc=bc, p=self.p)
        return phi[..., None, :]

    def value_tensor(self, uh: NDArray, bc: Tensor, index: Union[NDArray, slice]=np.s_[:]) -> Tensor:
        phi = self.basis_tensor(bc, index=index) # (NQ, NC, ldof)
        cell2dof = self.dof.cell_to_dof(index=index)
        uh_t = torch.tensor(uh, dtype=torch.float32)

        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        if self.doforder == 'nodes':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., NC)
            s1 = f"...ci, {s0[:dim]}ci->...{s0[:dim]}c"
            val = torch.einsum(s1, phi, uh_t[..., cell2dof])
        elif self.doforder == 'vdims':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ...)
            s1 = f"...ci, ci{s0[:dim]}->...c{s0[:dim]}"
            val = torch.einsum(s1, phi, uh_t[cell2dof, ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'nodes' and 'vdims'.")
        return val

# TODO: finish this

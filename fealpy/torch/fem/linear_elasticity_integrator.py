from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    CellOperatorIntegrator,
    enable_cache,
    assemblymethod,
    _S, Index, CoefLike
)

class LinearElasticityIntegrator(CellOperatorIntegrator):
    r"""The linear elasticity integrator for function spaces based on homogeneous meshes."""
    def __init__(self, 
                 lam: Optional[float]=None, mu: Optional[float]=None,
                 e: Optional[float]=None, nu: Optional[float]=None, 
                 coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        if lam is not None and mu is not None:
            self.e = mu * (3*lam + 2*mu) / (lam + mu)
            self.nu = lam / (2 * (lam + mu))
            self.lam = lam
            self.mu = mu
        elif e is not None and nu is not None:
            self.lam = nu * e / ((1 + nu) * (1 - 2 * nu))
            self.mu = e / (2 * (1 + nu))
            self.e = e
            self.nu = nu
        else:
            raise ValueError("Either (lam, mu) or (e, nu) should be provided.")
        self.coef = coef
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> Tensor:
        return space.cell_to_dof()[self.index]

    # @enable_cache
    # def fetch(self, space: _FS):
    #     q = self.q
    #     index = self.index
    #     mesh = getattr(space, 'mesh', None)
    #
    #     if not isinstance(mesh, HomogeneousMesh):
    #         raise RuntimeError("The LinearElasticityPlaneStrainOperatorIntegrator only support spaces on"
    #                            f"homogeneous meshes, but {type(mesh).__name__} is"
    #                            "not a subclass of HomoMesh.")
    #
    #     cm = mesh.entity_measure('cell', index=index)
    #     qf = mesh.integrator(q, 'cell')
    #     bcs, ws = qf.get_quadrature_points_and_weights()
    #     gphi = space.grad_basis(bcs, index=index, variable='x')
    #     return bcs, ws, gphi, cm, index
    
    def assembly(self, space: _FS) -> Tensor:
        e = self.e
        nu = self.nu
        D2_stress = e / (1 - nu**2) \
            * torch.tensor([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1 - nu) / 2]], dtype=torch.float64)
        #
        # I = torch.eye(2)
        # factor = e / (1 - nu**2)
        # D4_stress = factor * (
        #     (1 - nu) / 2 * (torch.einsum('ik,jl->ijkl', I, I) + torch.einsum('il,jk->ijkl', I, I)) +
        #     nu * torch.einsum('ij,kl->ijkl', I, I) +
        #     (1 - nu) * torch.einsum('ik,jl->ijkl', I, I)
        # )
        
        mu = self.mu
        lam = self.lam

        D2_strain = torch.tensor([[2 * mu + lam, lam, 0],
                                  [lam, 2 * mu + lam, 0],
                                  [0, 0, mu]], dtype=torch.float64)
        print("D2_strian:\n", D2_strain.shape, "\n", D2_strain)
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)
        cm = mesh.entity_measure('cell', index=index)
        print("cm:", cm.shape)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        print("ws:", ws.shape)
        B2_strain = space.strain(p=bcs, variable='x')
        print("B2_strain:", B2_strain.shape, "\n", B2_strain[0][0])
        K2_strain = torch.einsum('q, c, cqkj, kl, cqli-> cij',\
                        ws, cm, B2_strain, D2_strain, B2_strain)
        print("K2_strain:", K2_strain.shape, "\n", K2_strain[0])
        K2_stress = torch.einsum('q, c, cqkj, kl, cqli-> cij',\
                        ws, cm, B2_strain, D2_stress, B2_strain)
        print("K2_stress:", K2_stress.shape, "\n", K2_stress[0])




        # D2 = torch.tensor([
        #     [2 * mu + lam, lam, lam, 0, 0, 0],
        #     [lam, 2 * mu + lam, lam, 0, 0, 0],
        #     [lam, lam, 2 * mu + lam, 0, 0, 0],
        #     [0, 0, 0, mu, 0, 0],
        #     [0, 0, 0, 0, mu, 0],
        #     [0, 0, 0, 0, 0, mu]
        # ], dtype=torch.float64)



        #
        # GD = 3
        # I = torch.eye(GD)
        # D4 = mu * (torch.einsum('ik,jl->ijkl', I, I) \
        #           + torch.einsum('il,jk->ijkl', I, I)) \
        #     + lam * torch.einsum('ij,kl->ijkl', I, I)
        #
        # return D2_stress, D4_stress, D2_strain, D2, D4
        return K2_strain
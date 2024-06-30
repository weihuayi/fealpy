from typing import Optional

import numpy as np
from numpy.typing import NDArray

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

class LinearElasticityOperatorIntegrator(CellOperatorIntegrator):
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
    def to_global_dof(self, space: _FS) -> NDArray:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticityPlaneStrainOperatorIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        return bcs, ws, gphi, cm, index

    def assembly(self, space: _FS) -> NDArray:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(gphi, gphi, ws, cm, coef)
    
    def assembly1(self, space: _FS) -> NDArray:
        e = self.e
        nu = self.nu
        D2_stress = e / (1 - nu**2) \
            * np.array([[1, nu, 0],
                        [nu, 1, 0],
                        [0, 0, (1 - nu) / 2]], dtype=np.float64)
        
        I = np.eye(2)
        factor = e / (1 - nu**2)    
        D4_stress = factor * (
            (1 - nu) / 2 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I)) +
            nu * np.einsum('ij,kl->ijkl', I, I) +
            (1 - nu) * np.einsum('ik,jl->ijkl', I, I)
        )
        
        mu = self.mu
        lam  = self.lam

        D2_strain = np.array([[2 * mu + lam, lam, 0],
                              [lam, 2 * mu + lam, 0],
                              [0, 0, mu]], dtype=np.float64)

        D2 = np.array([
            [2 * mu + lam, lam, lam, 0, 0, 0],
            [lam, 2 * mu + lam, lam, 0, 0, 0],
            [lam, lam, 2 * mu + lam, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ], dtype=np.float64)

        GD = 3
        I = np.eye(GD)
        D4 = mu * (np.einsum('ik,jl->ijkl', I, I) \
                  + np.einsum('il,jk->ijkl', I, I)) \
            + lam * np.einsum('ij,kl->ijkl', I, I)
        
        return D2_stress, D4_stress, D2_strain, D2, D4
        
        


    
    @assemblymethod('fast')
    def fast_assembly(self, space: _FS) -> NDArray:
        """
        限制：常系数、单纯形网格
        TODO: 加入 assert
        """
        pass
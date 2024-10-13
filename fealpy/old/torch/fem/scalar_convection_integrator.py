#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: scalar_convection_integrator.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Fri 28 Jun 2024 04:50:24 PM CST
	@bref 
	@ref 
'''  
from typing import Optional

import torch
from torch import Tensor

from ..mesh import HomogeneousMesh
from ..utils import is_tensor
from ..functionspace.space import FunctionSpace as _FS
from .integrator import (
    CellOperatorIntegrator,
    enable_cache,
    assemblymethod,
    _S, Index, CoefLike
)
from ..utils import process_coef_func


class ScalarConvectionIntegrator(CellOperatorIntegrator):
    r"""The convection integrator for function spaces based on homogeneous meshes."""
    """
    @note ( coef \\cdot \\nabla u, v)
    """
    def __init__(self, coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None: 
        
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
    
    @enable_cache
    def to_global_dof(self, space: _FS) -> Tensor: 
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarConvectionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        phi = space.basis(bcs, index=index, variable='x')
        return bcs, ws, phi, gphi, cm, index
    
    def assembly(self, space: _FS) -> Tensor:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if is_tensor(coef):
            if self.batched:
                result = torch.einsum('q, cqk, cqmd, bcqd, c->bckm', ws, phi, gphi, coef.squeeze(0), cm)
            else:
                result = torch.einsum('q, cqk, cqmd, cqd, c->ckm', ws, phi, gphi, coef, cm)
        else:
            raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")
        return result

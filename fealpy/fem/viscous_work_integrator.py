#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: vector_viscous_work_integrator.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 17 Dec 2024 02:50:59 PM CST
	@bref 
	@ref 
'''  
from typing import Optional, Literal

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache
)


class ViscousWorkIntegrator(LinearInt, OpInt, CellInt):
    """
    construct the mu * (epslion(u), epslion(v)) fem matrix
    epsion(u) = 1/2*(\\nabla u+ (\\nabla u).T)
    """
    def __init__(self, coef: Optional[CoefLike] = None, q: Optional[int] = None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool = False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.set_region(region)
        self.batched = batched
    
    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        return space.cell_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch_assemble(self, space: _FS, /, indices=None):
        mesh = space.mesh
        index=self.entity_selection(indices)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index)
        cm = mesh.entity_measure('cell', index=index)
        return bcs, ws, gphi, cm

    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        coef = self.coef
        mesh = space.mesh
        bcs, ws, gphi, cm = self.fetch_assemble(space)
        index=self.entity_selection(indices)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        sym_gphi = 0.5*(bm.swapaxes(gphi, -2, -1) + gphi)
        result = bilinear_integral(sym_gphi, sym_gphi, ws, cm, coef, batched=self.batched)
        return result


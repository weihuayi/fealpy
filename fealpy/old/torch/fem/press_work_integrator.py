#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: press_work_integrator.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 02 Jul 2024 11:40:58 AM CST
	@bref 
	@ref 
'''  
from typing import Optional

from torch import Tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import CellOperatorIntegrator, _S, Index, CoefLike, enable_cache

class PressWorkIntegrator(CellOperatorIntegrator):
    def __init__(self, coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> Tensor:
        return space.cell_to_dof()[self.index]


    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        index = self.index
        q = self.q
        mesh = getattr(space[0], 'mesh', None)
        
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The PressWorkIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        phi = space0.basis(bcs, index=index, variable='x')
        gphi = space1.grad_basis(bcs, index=index, variable='x')
        return phi, gphi, cm, bcs, ws, index
    
    def assembly(self, space: _FS) -> Tensor:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        phi_0, gphi_1, cm, bcs, ws, index = self.fetch(space) 
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        result = bilinear_integral(gphi_1[...,0], phi_0, ws, cm, val, batched=self.batched)
        return result

class PressWorkIntegrator1(CellOperatorIntegrator):
    def __init__(self, coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> Tensor:
        return space.cell_to_dof()[self.index]


    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        index = self.index
        q = self.q
        mesh = getattr(space[0], 'mesh', None)
        
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The PressWorkIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        phi = space0.basis(bcs, index=index, variable='x')
        gphi = space1.grad_basis(bcs, index=index, variable='x')
        return phi, gphi, cm, bcs, ws, index
    
    def assembly(self, space: _FS) -> Tensor:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        phi_0, gphi_1, cm, bcs, ws, index = self.fetch(space) 
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        result = bilinear_integral(gphi_1[...,1], phi_0, ws, cm, val, batched=self.batched)
        return result


#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: press_work_integrator.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 15 Aug 2024 12:08:28 PM CST
	@bref 
	@ref 
'''  
from typing import Optional

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import LinearInt, OpInt, CellInt, CoefLike, enable_cache
from ..typing import TensorLike, Index, _S
from ..backend import backend_manager as bm


'''
(pI, \nabla v)
'''
class PressWorkIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        scalar_space = space[1].scalar_space
         
        index = self.index
        mesh = getattr(space[0], 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The PressWorkIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space0.basis(bcs, index=index)
        gphi = scalar_space.grad_basis(bcs ,index=index)
        if space1.dof_priority:
            gphi = gphi.swapaxes(-1,-2)
            gphi = gphi.reshape(*gphi.shape[:2], -1) 
        else:
            gphi = gphi.reshape(*gphi.shape[:2], -1)
        return phi, gphi, cm, bcs, ws, index

    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        phi, gphi, cm, bcs, ws, index = self.fetch(space) 
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        result = bilinear_integral(gphi, phi, ws, cm, val, batched=self.batched)
        print(result.shape)
        return result


class PressWorkIntegrator0(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        index = self.index
        mesh = getattr(space[0], 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The PressWorkIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space0.basis(bcs, index=index)
        gphi = space1.grad_basis(bcs, index=index)
        return phi, gphi, cm, bcs, ws, index

    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        phi_0, gphi_1, cm, bcs, ws, index = self.fetch(space) 
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        result = bilinear_integral(gphi_1[...,0], phi_0, ws, cm, val, batched=self.batched)
        return result


class PressWorkIntegrator1(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        index = self.index
        mesh = getattr(space[0], 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The PressWorkIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space0.basis(bcs, index=index)
        gphi = space1.grad_basis(bcs, index=index)
        return phi, gphi, cm, bcs, ws, index

    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        phi_0, gphi_1, cm, bcs, ws, index = self.fetch(space) 
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        result = bilinear_integral(gphi_1[...,1], phi_0, ws, cm, val, batched=self.batched)
        return result

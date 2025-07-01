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
from .integrator import LinearInt, OpInt, CellInt, FaceInt, enable_cache
from ..typing import Threshold, TensorLike, Index, _S, CoefLike 
from ..backend import backend_manager as bm
from ..functionspace import TensorFunctionSpace

'''
(p, \\nabla \\cdot v)
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
        return (space[0].cell_to_dof()[self.index],
                space[1].cell_to_dof()[self.index])

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
        q = space[0].p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space0.basis(bcs, index=index)
        gphi = space1.grad_basis(bcs ,index=index)
        return phi, gphi, cm, bcs, ws, index

    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        phi, gphi, cm, bcs, ws, index = self.fetch(space) 
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if isinstance(space[0], TensorFunctionSpace):
            gphi = gphi
        else:
            gphi = bm.einsum('...ii->...', gphi)
        result = bilinear_integral(gphi, phi, ws, cm, val, batched=self.batched)
        return result

class BoundaryPressWorkIntegrator(LinearInt, OpInt, FaceInt):
    '''
    (pn, v)_\\{\\partial \\Omega\\}
    '''
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        self.coef = coef
        self.q = q
        self.threshold = threshold
        self.batched = batched

    def make_index(self, space: _FS) -> TensorLike:
        threshold = self.threshold
        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            mesh = space[0].mesh
            index = mesh.boundary_face_index()
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]
        return index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        tag = space[0].mesh.face2cell[index,0]
        result0 = space[0].cell_to_dof()[tag]
        result1 = space[1].cell_to_dof()[tag]
        return (result0, result1) 
    
    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        index = self.make_index(space)
        mesh = space[0].mesh

        facemeasure = mesh.entity_measure('face', index=index)
        q = space[0].p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        p_phi = space[0].cell_basis_on_face(bcs, index)
        u_phi = space[1].cell_basis_on_face(bcs, index)
        n = mesh.face_unit_normal(index)
        return bcs, ws, p_phi, u_phi, facemeasure, index, n

    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)
        bcs, ws, p_phi, u_phi, fm, index, n = self.fetch(space)
        p_phi = bm.einsum('cql, cd->cqld', p_phi, n)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        result = bilinear_integral(u_phi, p_phi, ws, fm, val, batched=self.batched)
        return result

class PressWorkIntegratorX(LinearInt, OpInt, CellInt):
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


class PressWorkIntegratorY(LinearInt, OpInt, CellInt):
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

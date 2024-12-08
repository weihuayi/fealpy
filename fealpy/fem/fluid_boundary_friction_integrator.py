#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: fluid_boundary_friction_integrator.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 05 Dec 2024 02:58:11 PM CST
	@bref 
	@ref 
'''  
from typing import Optional
from fealpy.backend import backend_manager as bm
from ..typing import TensorLike, SourceLike, Threshold
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    LinearInt, OpInt, FaceInt,
    enable_cache,
    assemblymethod,
    CoefLike
)
'''
@brief
(coef \\nabla u \\cdot n, v)_{\\partial \\Omega}
@param[in] mu 
'''
class FluidBoundaryFrictionIntegrator(LinearInt, OpInt, FaceInt):
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
            mesh = space.mesh
            index = mesh.boundary_face_index()
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]
        return index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        result1 = space.face_to_dof(index=index) 
        tag = space.mesh.edge2cell[:,0]
        result2 = space.cell_to_dof()[tag]
        return (result1, result2) 
    
    @enable_cache
    def fetch(self, space: _FS):
        index = self.make_index(space)
        mesh = space.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarRobinBCIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        facemeasure = mesh.entity_measure('face', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        edge2cell = mesh.face_to_cell()
        phi = space.cell_basis_on_edge(bcs, edge2cell[index,2])
        gphi = space.cell_grad_basis_on_edge(bcs, edge2cell[index,0], edge2cell[index,2])
        n = mesh.edge_normal(index)
        return bcs, ws, phi, gphi, facemeasure, index, n

    def assembly(self, space: _FS):
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, fm, index, n = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        gphin = bm.einsum('ej, eqlj->eql', n, gphi)
        result =  bilinear_integral(phi, gphin, ws, fm, val, batched=self.batched)
        return result

class FluidBoundaryFrictionIntegratorP(LinearInt, OpInt, FaceInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False, mesh):
        super().__init__()
        self.coef = coef
        self.q = q
        self.threshold = threshold
        self.batched = batched
        self.mesh = mesh
   
    def make_index(self, space: _FS) -> TensorLike:
        threshold = self.threshold
        mesh = self.mesh
        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            index = mesh.boundary_face_index()
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]
        return index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        return space.face_to_dof(index=index)
    
    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        index = self.make_index(space)
        mesh = self.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarRobinBCIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        facemeasure = mesh.entity_measure('face', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi1 = space[1].face_basis(bcs, index)
        phi0 = space[0].face_basis(bcs, index)
        n = mesh.edge_normal(index)
        return bcs, ws, phi0, phi1, facemeasure, index, n

    def assembly(self, space: _FS):
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi0, phi1, fm, index, n = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        phi0n = bm.einsum('ej, e...j->e...j', n, phi0)
        result =  bilinear_integral(phi1, phi0n, ws, fm, val, batched=self.batched)
        return result

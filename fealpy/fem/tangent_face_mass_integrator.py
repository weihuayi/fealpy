from fealpy.backend import backend_manager as bm

from typing import Optional
from fealpy.typing import TensorLike, SourceLike, Threshold
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import bilinear_integral

from fealpy.fem.integrator import (
    LinearInt, OpInt, FaceInt,
    enable_cache
)

class TangentFaceMassIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, coef=None, q: Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        self.coef = coef
        self.q = q
        self.threshold = threshold
        self.batched = batched

    def make_index(self, space: _FS):
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
        return space.face_to_dof()[index]
    
    @enable_cache
    def fetch(self, space: _FS):
        index = self.make_index(space)
        mesh = space.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarRobinBCIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        facemeasure = mesh.entity_measure('edge', index=index)
        t = mesh.edge_unit_tangent()
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs, index)
        return bcs, ws, phi, facemeasure, index, t
    
    def assembly(self, space: _FS):
        coef = self.coef
        bcs, ws, phi, fm, index, t = self.fetch(space)
        mesh = space.mesh
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        t[...,0] = 1
        phit = bm.einsum('eqid, ed -> eqi', phi, t[index,...])
        phii = bm.einsum('eqid -> eqi', phi)
        result = bilinear_integral(phii, phit, ws, fm, val, batched=self.batched)
        return result

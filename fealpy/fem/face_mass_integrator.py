from typing import Optional

from ..typing import TensorLike, Threshold, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import LinearInt, OpInt, FaceInt, enable_cache


class _FaceMassIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        self.coef = coef
        self.q = q
        self.threshold = threshold
        self.batched = batched

    
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        return space.face_to_dof(index=index)
    
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
        phi = space.face_basis(bcs,index=index)
        return bcs, ws, phi, facemeasure, index
    
    def assembly(self, space: _FS):
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, fm, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        return bilinear_integral(phi, phi, ws, fm, val, batched=self.batched)

class InterFaceMassIntegrator(_FaceMassIntegrator):
    def make_index(self, space: _FS):
        index = self.threshold
        return index

class BoundaryFaceMassIntegrator(_FaceMassIntegrator):
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

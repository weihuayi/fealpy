from typing import Optional

from ..typing import TensorLike, SourceLike, Threshold
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..functional import linear_integral
from .integrator import LinearInt, SrcInt, FaceInt, enable_cache
from ..utils import process_coef_func


class _FaceSourceIntegrator(LinearInt, SrcInt, FaceInt):
    def __init__(self, source: Optional[SourceLike]=None, q:Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        self.source = source 
        self.q = q
        self.threshold = threshold
        self.batched = batched


    @enable_cache
    def to_global_dof(self, space) -> TensorLike:
        index = self.make_index(space)
        return space.face_to_dof(index=index)

    @enable_cache
    def fetch(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        mesh = space.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        facemeasure = mesh.entity_measure('face', index=index)

        n = mesh.face_unit_normal(index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs) # just for simplex mesh, TODO: consider othter type mesh

        return bcs, ws, phi, facemeasure, index, n

    def assembly(self, space):
        source = self.source
        bcs, ws, phi, fm, index, n = self.fetch(space) 
        mesh = getattr(space, 'mesh', None)
        val = process_coef_func(source, bcs=bcs, mesh=mesh, etype='face', index=index, n=n)
        return linear_integral(phi, ws, fm, val, self.batched)

class InterFaceSourceIntegrator(_FaceSourceIntegrator):
    def make_index(self, space: _FS):
        index = self.threshold
        return index

class BoundaryFaceSourceIntegrator(_FaceSourceIntegrator): 
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

from typing import Optional

from ..typing import TensorLike, SourceLike, Threshold
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..functional import linear_integral
from .integrator import LinearInt, SrcInt, FaceInt, enable_cache


class ScalarInterfaceBCIntegrator(LinearInt, SrcInt, FaceInt):
    def __init__(self, source: SourceLike, q:Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        self.source = source 
        self.q = q
        self.threshold = threshold
        self.batched = batched

    @enable_cache
    def make_index(self, space: _FS):
        index = self.threshold
        return index

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

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs) # just for simplex mesh, TODO: consider othter type mesh

        return bcs, ws, phi, facemeasure

    def assembly(self, space):
        source = self.source
        index = self.make_index(space)
        bcs, ws, phi, fm = self.fetch(space)

        if callable(source):
            if (not hasattr(source, 'coordtype')) or (source.coordtype == 'cartesian'):
                mesh = space.mesh
                ps = mesh.bc_to_point(bcs, index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算 Neumann 边界的法向梯度
                val = source(ps)
            elif gN.coordtype == 'barycentric':
                # 这个时候 gN 是一个有限元函数，一定不需要算面法向
                val = source(bcs, index=index)
        else:
            val = source 

        return linear_integral(phi, ws, fm, val, self.batched)

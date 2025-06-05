
from ...backend import TensorLike as _DT
from ...backend import backend_manager as bm


class CosCosData():
    description = ""

    def geo_dimension(self) -> int:
        return 2

    def domain(self):
        return [-1., 1., -1., 1.]

    def init_mesh(self):
        from ...mesh import TriangleMesh
        return TriangleMesh.from_box(self.domain())

    def solution(self, p: _DT) -> _DT:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.cos(pi*x)*bm.cos(pi*y)
        return val # val.shape == x.shape

    def gradient(self, p: _DT) -> _DT:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -pi*bm.sin(pi*x)*bm.cos(pi*y),
            -pi*bm.cos(pi*x)*bm.sin(pi*y)), axis=-1)
        return val # val.shape == p.shape

    def source(self, p: _DT) -> _DT:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 2*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)
        return val

    def dirichlet(self, p: _DT) -> _DT:
        return self.solution(p)
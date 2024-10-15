
import torch
from fealpy.ml.sampler import (
    ISampler,
    BoxBoundarySampler,
    MeshSampler
)
from fealpy.ml.sampler.sampler import _PolytopeSampler, _QuadSampler

def _valified_range(t: torch.Tensor, low, high):
    assert torch.all(torch.all(t >= low, dim=0))
    assert torch.all(torch.all(t <= high, dim=0))

class TestSimple():
    def test_isampler(self):
        s = ISampler([[0, 1], [1, 3]])
        assert s.nd == 2

        out = s.run(50)
        assert out.shape == (50, 2)
        _valified_range(out[:, 0], 0, 1)
        _valified_range(out[:, 1], 1, 3)


    def test_boxboundarysampler(self):
        s = BoxBoundarySampler([0, 0, 0], [1, 1, 1])
        assert s.nd == 3

        out = s.run(50)
        assert out.shape == (300, 3)
        _valified_range(out, 0, 1)


    def test_trisampler(self):
        from fealpy.mesh import TriangleMesh

        mesh = TriangleMesh.from_box([0, 2, 0, 3], nx=10, ny=10)
        s = MeshSampler(mesh, 'cell')
        assert isinstance(s, _PolytopeSampler)
        assert s.nd == 2

        out = s.run(10)
        assert out.shape == (2000, 2)
        _valified_range(out[:, 0], 0, 2)
        _valified_range(out[:, 1], 0, 3)


    def test_tetsampler(self):
        from fealpy.mesh import TetrahedronMesh

        mesh = TetrahedronMesh.from_box([0, 1, 1, 3, 2, 5], nx=5, ny=5, nz=5)
        s = MeshSampler(mesh, 'cell')
        assert isinstance(s, _PolytopeSampler)
        assert s.nd == 3

        out = s.run(10)
        assert out.shape == (7500, 3)
        _valified_range(out[:, 0], 0, 1)
        _valified_range(out[:, 1], 1, 3)
        _valified_range(out[:, 2], 2, 5)


    def test_quadsampler(self):
        from fealpy.mesh import QuadrangleMesh

        mesh = QuadrangleMesh.from_box([0, 1, 1, 2], nx=10, ny=10)
        s = MeshSampler(mesh, 'cell')
        assert isinstance(s, _QuadSampler)
        assert s.nd == 2

        out = s.run(10)
        assert out.shape == (1000, 2)
        _valified_range(out[:, 0], 0, 1)
        _valified_range(out[:, 1], 1, 2)

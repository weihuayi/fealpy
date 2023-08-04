
import torch
from fealpy.pinn.sampler import (
    ISampler,
    BoxBoundarySampler,
    get_mesh_sampler,
    TMeshSampler,
    QuadrangleMeshSampler
)
from fealpy.pinn.sampler.sampler import JoinedSampler, HybridSampler


def _valified_range(t: torch.Tensor, low, high):
    assert torch.all(torch.all(t >= low, dim=0))
    assert torch.all(torch.all(t <= high, dim=0))

class TestSimple():
    def test_isampler(self):
        s = ISampler(50, [[0, 1], [1, 3]])
        assert s.m == 50
        assert s.nd == 2

        out = s.run()
        assert out.shape == (50, 2)
        _valified_range(out[:, 0], 0, 1)
        _valified_range(out[:, 1], 1, 3)


    def test_boxboundarysampler(self):
        s = BoxBoundarySampler(50, [0, 0, 0], [1, 1, 1])
        assert s.m == 300
        assert s.nd == 3

        out = s.run()
        assert out.shape == (300, 3)
        _valified_range(out, 0, 1)


    def test_trisampler(self):
        from fealpy.mesh import TriangleMesh

        mesh = TriangleMesh.from_box([0, 2, 0, 3], nx=10, ny=10)
        s = get_mesh_sampler(10, mesh)
        assert isinstance(s, TMeshSampler)
        assert s.m == 2000
        assert s.nd == 2

        out = s.run()
        assert out.shape == (2000, 2)
        _valified_range(out[:, 0], 0, 2)
        _valified_range(out[:, 1], 0, 3)

        assert s.weight.shape == (2000, 1)


    def test_tetsampler(self):
        from fealpy.mesh import TetrahedronMesh

        mesh = TetrahedronMesh.from_box([0, 1, 1, 3, 2, 5], nx=5, ny=5, nz=5)
        s = get_mesh_sampler(10, mesh)
        assert isinstance(s, TMeshSampler)
        assert s.m == 7500
        assert s.nd == 3

        out = s.run()
        assert out.shape == (7500, 3)
        _valified_range(out[:, 0], 0, 1)
        _valified_range(out[:, 1], 1, 3)
        _valified_range(out[:, 2], 2, 5)

        assert s.weight.shape == (7500, 1)


    def test_quadsampler(self):
        from fealpy.mesh import QuadrangleMesh

        mesh = QuadrangleMesh.from_box([0, 1, 1, 2], nx=10, ny=10)
        s = get_mesh_sampler(10, mesh)
        assert isinstance(s, QuadrangleMeshSampler)
        assert s.m == 1000
        assert s.nd == 2

        out = s.run()
        assert out.shape == (1000, 2)
        _valified_range(out[:, 0], 0, 1)
        _valified_range(out[:, 1], 1, 2)

        assert s.weight.shape == (1000, 1)


class TestCombiation():
    def test_joined(self):
        s1 = ISampler(10, [[1, 3]])
        s2 = ISampler(11, [[2, 4]])
        s = JoinedSampler(s1, s2)
        s3 = ISampler(5, [[5, 6]])
        s.add(s3)
        assert s.m == 26
        assert s.nd == 1

        out = s.run()
        assert out.shape == (26, 1)


    def test_hybrid(self):
        s1 = ISampler(10, [[1, 3]])
        s2 = ISampler(10, [[2, 4], [8, 9]])
        s = HybridSampler(s1, s2)
        s3 = ISampler(10, [[5, 6], [4, 5], [-1, 0]])
        s.add(s3)
        assert s.m == 10
        assert s.nd == 6

        out = s.run()
        assert out.shape == (10, 6)


    def test_multi_1(self):
        s = ISampler(5, [[1, 2]]) & ISampler(6, [[2, 3]])
        s = ISampler(11, [[3, 4]]) | s
        assert s.m == 11
        assert s.nd == 2

        out = s.run()
        assert out.shape == (11, 2)


    def test_multi_2(self):
        s = ISampler(5, [[1, 2], [5, 6]]) | ISampler(5, [[2, 3]])
        s = ISampler(10, [[3, 4], [6, 8], [1, 7]]) & s
        assert s.m == 15
        assert s.nd == 3

        out = s.run()
        assert out.shape == (15, 3)

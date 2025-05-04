import pytest
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh
from fealpy.mesh import UniformMesh1d
from fealpy.mesh import UniformMesh2d
from fealpy.mesh import UniformMesh3d

class TestUniformMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("domain, extent, expected_cls, expected_h, expected_origin", [
        # 1D positional domain+extent
        ([0.0, 1.0], [0, 10], UniformMesh1d, (0.1,), (0.0,)),
        # 2D keyword domain+extent
        ([0.0, 2.0, -1.0, 3.0], [0, 4, 2, 6], UniformMesh2d, (0.5, 1.0), (0.0, -1.0)),
        # 3D positional domain+extent
        ([0.0, 1.0, 0.0, 1.0, 0.0, 2.0], [0, 2, 0, 2, 1, 3], UniformMesh3d, (0.5, 0.5, 1.0), (0.0, 0.0, 0.0)),
    ])
    def test_domain_extent_overload(self, domain, extent, expected_cls,
                                    expected_h, expected_origin, backend):
        # Test overload (domain, extent) both positional and keyword
        bm.set_backend(backend)
        mesh_pos = UniformMesh(domain, extent) if False else UniformMesh(domain, extent)
        mesh_kw = UniformMesh(domain=domain, extent=extent)
        for mesh in (mesh_pos, mesh_kw):
            assert isinstance(mesh, expected_cls)
            assert mesh.h == expected_h
            assert mesh.origin == expected_origin
            assert mesh.extent == tuple(extent)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("args, kwargs, expected_cls, expected_h, expected_origin", [
        # 1D positional extent+h+origin
        (((0, 5), 0.5, 1.0), {}, UniformMesh1d, (0.5,), (1.0,)),
        # 2D keyword extent+h+origin
        ((), {"extent": (0, 3, 1, 5), "h": (1.0, 2.0), "origin": (0.0, 1.0)}, UniformMesh2d, (1.0, 2.0), (0.0, 1.0)),
        # 3D positional extent+h+origin
        (((0, 2, 0, 2, 1, 3), (0.5, 0.5, 2.0), (0.0, 0.0, 1.0)), {}, UniformMesh3d, (0.5, 0.5, 2.0), (0.0, 0.0, 1.0)),
    ])
    def test_extent_h_origin_overload(self, args, kwargs, expected_cls,
                                      expected_h, expected_origin, backend):
        # Test overload (extent, h, origin) both positional and keyword
        bm.set_backend(backend)
        mesh_pos = UniformMesh(*args) if args else None
        mesh_kw = UniformMesh(**kwargs) if kwargs else mesh_pos
        for mesh in (mesh_pos, mesh_kw):
            assert isinstance(mesh, expected_cls)
            assert mesh.h == expected_h
            assert mesh.origin == expected_origin
            assert mesh.extent == tuple(args[0] if args else kwargs["extent"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("bad_args, bad_kwargs", [
        # wrong number of positional args
        ((1, 2, 3, 4), {}),
        # missing extent keyword
        ((), {"domain": [0.0, 1.0]}),
        # domain/extent length mismatch
        ((), {"domain": [0.0, 1.0, 2.0], "extent": [0, 2, 0, 2]}),
        # zero cells (imin==imax)
        ([0.0, 1.0], [5, 5]),
        # h length mismatch
        ((), {"extent": (0, 5, 0, 5), "h": (0.2,), "origin": (0.0, 0.0)}),
        # origin length mismatch
        ((), {"extent": (0, 5), "h": 0.5, "origin": (0.0, 1.0)}),
        # extent non-int
        ((), {"extent": (0.0, 5.0), "h": 1.0, "origin": 0.0}),
    ])
    def test_invalid_inputs(self, bad_args, bad_kwargs, backend):
        # Test invalid argument combinations raise ValueError
        bm.set_backend(backend)
        with pytest.raises(ValueError):
            if bad_args:
                UniformMesh(*bad_args)
            else:
                UniformMesh(**bad_kwargs)

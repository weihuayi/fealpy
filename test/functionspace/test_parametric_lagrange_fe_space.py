import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh
from fealpy.geometry.implicit_surface import SphereSurface 
from fealpy.functionspace import LagrangeFESpace, ParametricLagrangeFESpace

from parametric_lagrange_fe_space_data import *


class TestParametricFESpace:

    @pytest.mark.parametrize("backend", ['numpy', 'jax'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_interpolate(self, meshdata,  backend):
        bm.set_backend(backend)

        p = 2
        lmesh = TriangleMesh.from_unit_sphere_surface()
        surface = SphereSurface()
        mesh = LagrangeTriangleMesh.from_triangle_mesh(lmesh, p=p, surface=surface)
        space = ParametricLagrangeFESpace(mesh, p=p)
        
        def fun(p):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            return x**2+y**2+z**2-1
        interpolate = space.interpolate(u = fun)

        np.testing.assert_allclose(bm.to_numpy(interpolate), meshdata["interpolate"],1e-15)

    @pytest.mark.parametrize("backend", ['numpy', 'jax'])
    def test_boundary_interpolate(self, backend):
        bm.set_backend(backend)
        def solution(p):
            x = p[..., 0]
            y = p[..., 1]
            return x+y

        p = 2
        mesh = TriangleMesh.from_box([0, 1, 0, 1], 1, 1)
        lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=p)
        space = ParametricLagrangeFESpace(lmesh, p=p)
        uh = bm.zeros(space.number_of_global_dofs())
        isbdof = lmesh.boundary_node_flag()
        bD = space.boundary_interpolate(gD=solution, uh=uh)

if __name__ == "__main__":
    a = TestParametricFESpace()
    #a.test_interpolate(init_data[0], 'numpy')
    a.test_boundary_interpolate('numpy')
    #pytest.main(["./test_lagrange_triangle_mesh.py"])

import numpy as np
import sympy as sp

import pytest
from fealpy.pde.surface_poisson_model import SurfaceLevelSetPDEData
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.mesh.lagrange_quadrangle_mesh import LagrangeQuadrangleMesh
from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace

from lagrange_quadrangle_mesh_data import *


class TestLagrangeQuadrangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_surface_mesh(self, backend):
        bm.set_backend(backend)

        surface = SphereSurface()
        mesh = QuadrangleMesh.from_unit_sphere_surface()

        lmesh = LagrangeQuadrangleMesh.from_quadrangle_mesh(mesh, p=1, surface=surface)
        fname = f"sphere_qtest.vtu"
        lmesh.to_vtk(fname=fname)

    """
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", from_quadrangle_mesh_data)
    def test_from_quadranglemesh(self, data, backend):
        bm.set_backend(backend)

        p = data['p']
        surface = data['surface']
        mesh = QuadrangleMesh.from_unit_sphere_surface()

        lmesh = LagrangeQuadrangleMesh.from_triangle_mesh(mesh, p, surface=surface)

        assert lmesh.number_of_nodes() == data["NN"] 
        assert lmesh.number_of_edges() == data["NE"] 
        assert lmesh.number_of_faces() == data["NF"] 
        assert lmesh.number_of_cells() == data["NC"] 
        
        cell = lmesh.entity('cell')
        np.testing.assert_allclose(bm.to_numpy(cell), data["cell"], atol=1e-14)
        """


if __name__ == "__main__":
    a = TestLagrangeQuadrangleMeshInterfaces()
    a.test_surface_mesh('numpy')
    #a.test_from_quadrangle_mesh(from_quadrangle_mesh_data[0], 'numpy')
    #pytest.main(["./test_lagrange_triangle_mesh.py"])

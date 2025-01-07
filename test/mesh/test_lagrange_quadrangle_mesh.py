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

        lmesh = LagrangeQuadrangleMesh.from_quadrangle_mesh(mesh, p=3, surface=surface)
        fname = f"sphere_qtest.vtu"
        lmesh.to_vtk(fname=fname)

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", cell_area_data)
    def test_cell_area(self, data, backend):
        bm.set_backend(backend)

        surface = SphereSurface() #以原点为球心，1 为半径的球
        mesh = QuadrangleMesh.from_unit_sphere_surface()
        
        # 计算收敛阶
        p = 2
        maxit = 4
        cm = np.zeros(maxit, dtype=np.float64)
        em = np.zeros(maxit, dtype=np.float64)
        for i in range(maxit):
            lmesh = LagrangeQuadrangleMesh.from_quadrangle_mesh(mesh, p=p, surface=surface)
        
            cm[i] = np.sum(lmesh.cell_area())
            
            x = bm.to_numpy(cm[i])
            y = data["sphere_cm"]
            em[i] = np.abs(x - y)  # absolute error

            if i < maxit-1:
                mesh.uniform_refine()
            
        em_ratio = em[0:-1] / em[1:]
        print("unit_sphere:", em_ratio)

if __name__ == "__main__":
    a = TestLagrangeQuadrangleMeshInterfaces()
    a.test_surface_mesh('numpy')
    #a.test_cell_area(cell_area_data[0], 'numpy')
    #pytest.main(["./test_lagrange_triangle_mesh.py"])

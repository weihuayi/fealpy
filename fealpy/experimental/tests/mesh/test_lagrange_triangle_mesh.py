import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.geometry import SphereSurface, EllipsoidSurface
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.experimental.tests.mesh.lagrange_triangle_mesh_data import *

class TestLagrangeTriangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", init_data)
    def test_init_mesh(self, data, backend):
        bm.set_backend(backend)

        p = data['p']
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])
        surface = data['surface']

        #ipdb.set_trace()
        mesh = LagrangeTriangleMesh(node, cell, p, surface=surface, construct=True)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"]
        assert mesh.number_of_faces() == data["NF"]
        assert mesh.number_of_cells() == data["NC"] 

        cell = mesh.entity('cell')
        np.testing.assert_allclose(bm.to_numpy(cell), data["cell"], atol=1e-14)   

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", from_triangle_mesh_data)
    def test_from_triangle_mesh(self, data, backend):
        bm.set_backend(backend)

        p = data['p']
        surface = data['surface']
        mesh = TriangleMesh.from_unit_sphere_surface()

        lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p, surface=surface)

        assert lmesh.number_of_nodes() == data["NN"] 
        assert lmesh.number_of_edges() == data["NE"] 
        assert lmesh.number_of_faces() == data["NF"] 
        assert lmesh.number_of_cells() == data["NC"] 
        
        cell = lmesh.entity('cell')
        np.testing.assert_allclose(bm.to_numpy(cell), data["cell"], atol=1e-14)   

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_surface_mesh(self, backend):
        bm.set_backend(backend)

        surface = SphereSurface()
        mesh = TriangleMesh.from_unit_sphere_surface()

        lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=3, surface=surface)
        fname = f"sphere_test.vtu"
        lmesh.to_vtk(fname=fname)
        
        ellsurface = EllipsoidSurface()
        ellmesh = TriangleMesh.from_ellipsoid(radius=[5,5,3])

        elmesh = LagrangeTriangleMesh.from_triangle_mesh(ellmesh, p=3, surface=ellsurface)
        fname = f"elip_test.vtu"
        elmesh.to_vtk(fname=fname)

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", cell_area_data)
    def test_cell_area(self, data, backend):
        bm.set_backend(backend)

        surface = SphereSurface() #以原点为球心，1 为半径的球
        mesh = TriangleMesh.from_unit_sphere_surface()

        # 计算收敛阶 
        maxit = 4
        cm = np.zeros(maxit, dtype=np.float64)
        em = np.zeros(maxit, dtype=np.float64)
        for i in range(maxit):
            lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=3, surface=surface)
        
            cm[i] = np.sum(lmesh.cell_area())
            
            x = bm.to_numpy(cm[i])
            y = data["sphere_cm"]
            em[i] = np.abs(x - y)  # absolute error

            if i < maxit-1:
                mesh.uniform_refine()
            
        em_ratio = em[0:-1] / em[1:]
        print("unit_sphere:", em_ratio)


    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", edge_length_data)
    def test_edge_length(self, data, backend):
        bm.set_backend(backend)

        surface = SphereSurface() #以原点为球心，1 为半径的球
        mesh = TriangleMesh.from_unit_sphere_surface()
        lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=3, surface=surface)
        el = lmesh.edge_length()        
       
        np.testing.assert_allclose(bm.to_numpy(el), data["el"], atol=1e-14)   

if __name__ == "__main__":
    #a = TestLagrangeTriangleMeshInterfaces()
    #a.test_init_mesh(init_data[0], 'numpy')
    #a.test_from_triangle_mesh(from_triangle_mesh_data[0], 'numpy')
    #a.test_surface_mesh('numpy')
    #a.test_cell_area(cell_area_data[0], 'numpy')
    #a.test_edge_length(edge_length_data[0], 'numpy')
    #a.test_(cell_[0], 'numpy')
    pytest.main(["./test_lagrange_triangle_mesh.py"])

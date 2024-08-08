import ipdb
import numpy as np
import matplotlib.pyplot as plt

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.tetrahedron_mesh import TetrahedronMesh 
from fealpy.experimental.tests.mesh.tetrahedron_mesh_data import *


class TestTetrahedronMeshInterfaces:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])

        mesh = TetrahedronMesh(node, cell)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
        
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", from_one_tetrahedron_data)
    def test_from_one_tetrahedron(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_one_tetrahedron(meshtype=data['meshtype'])

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", face_to_edge_sign_data)
    def test_face_to_edge_sign(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
        sign = mesh.face_to_edge_sign() 
        np.testing.assert_array_equal(bm.to_numpy(sign), data["sign"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", face_unit_norm)
    def test_face_unit_norm(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        n = mesh.face_unit_normal()
        np.testing.assert_allclose(bm.to_numpy(n), data["fn"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", from_box)
    def test_from_box(self, data, backend):
        bm.set_backend(backend)
        threshold = data["threshold"]
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1, threshold=threshold)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(edge), data["edge"])
        np.testing.assert_array_equal(bm.to_numpy(face), data["face"])
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        face2cell = mesh.face_to_cell()
        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", entity_measure)
    def test_entity_measure(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        nm = mesh.entity_measure(etype=0)
        em = mesh.entity_measure(etype=1) 
        fm = mesh.entity_measure(etype=2)
        cm = mesh.entity_measure(etype=3) 

        np.testing.assert_allclose(bm.to_numpy(nm), data["nm"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(em), data["em"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(fm), data["fm"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(cm), data["cm"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", grad_lambda)
    def test_grad_lambda(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        glambda = mesh.grad_lambda()
        np.testing.assert_allclose(bm.to_numpy(glambda), data["glambda"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", grad_shape_function)
    def test_grad_shape_function(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=1,ny=1,nz=1)
        qf = mesh.integrator(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gsf = mesh.grad_shape_function(bcs, p=2, variables=data['variables'])

        np.testing.assert_allclose(bm.to_numpy(gsf), data["grad_shape_function"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", number_of_local_ipoints)
    def test_number_of_local_ipoints(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
     
        ldof = mesh.number_of_local_ipoints(p=3,iptype=data["iptype"])
        assert ldof == data["ldof"] 

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", number_of_global_ipoints)
    def test_number_of_global_ipoints(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
     
        gdof = mesh.number_of_global_ipoints(p=4)
        assert gdof == data["gdof"] 

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", interpolation_points)
    def test_interpolation_points(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        ipoint = mesh.interpolation_points(p=4)
        np.testing.assert_allclose(bm.to_numpy(ipoint), data["ipoint"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", face_to_ipoint)
    def test_face_to_ipoint(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        face2ipoint = mesh.face_to_ipoint(p=4)
        np.testing.assert_allclose(bm.to_numpy(face2ipoint), data["f2p"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", cell_to_ipoint)
    def test_cell_to_ipoint(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        cell2ipoint = mesh.cell_to_ipoint(p=4)
        np.testing.assert_allclose(bm.to_numpy(cell2ipoint), data["cell2ipoint"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", face_unit_normal)
    def test_face_normal(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=1,ny=1,nz=1)
        fn = mesh.face_unit_normal()
        np.testing.assert_allclose(bm.to_numpy(fn), data["fn"], atol=1e-14)


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", uniform_refine)
    def test_unifrom_refine(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=1,ny=1,nz=1)
        mesh.uniform_refine(n=2)
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", circumcenter)
    def test_circumcenter(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        c = mesh.circumcenter()
        np.testing.assert_allclose(bm.to_numpy(c), data["c"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", from_unit_sphere_gmsh)
    def test_from_unit_sphere_gmsh(self, data, backend):
        bm.set_backend(backend)
        sphere = TetrahedronMesh.from_unit_sphere_gmsh(1)
        node = sphere.entity('node')
        cell = sphere.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = sphere.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", from_unit_cube)
    def test_from_unit_cube(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_unit_cube(3,2,1)
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", from_cylinder_gmsh)
    def test_from_cylinder_gmsh(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_cylinder_gmsh(1,1,1)
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])




if __name__ == "__main__":
    #pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_init"])
    #pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_from_one_tetrahedron"])
    #pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_from_face_to_edge_sign"])
    #a = TestTetrahedronMeshInterfaces()
    #a.test_face_unit_norm(face_unit_norm[0], 'pytorch')
    #pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_face_unit_norm"])
    #pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_from_box"])
    #pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_entity_measure"])
    pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_grad_lambda"])

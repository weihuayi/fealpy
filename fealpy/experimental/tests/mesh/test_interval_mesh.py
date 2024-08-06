import ipdb
import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.interval_mesh import IntervalMesh
from fealpy.experimental.tests.mesh.interval_mesh_data import *  
from fealpy.experimental.mesh import TriangleMesh

from fealpy.mesh import IntervalMesh as IMesh
from fealpy.mesh import TriangleMesh as TMesh
import matplotlib.pyplot as plt
import numpy as np


class TestIntervalMeshInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self,meshdata,backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = IntervalMesh(node,cell)

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]

        face2cell = mesh.face_to_cell()

        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

        cell2edge = mesh.cell_to_edge()
        print('cell2edge:',cell2edge)
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", from_interval_domain_data)
    def test_from_interval_domain(self,meshdata ,backend):
        interval = bm.from_numpy(meshdata['interval'])
        n = meshdata['n']
        mesh = IntervalMesh.from_interval_domain(interval , n)

        node = mesh.node
        cell = mesh.cell

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]

        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"])
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

        cell2edge = mesh.cell_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", from_mesh_boundary_data)
    def test_from_mesh_boundary(self,meshdata ,backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = IntervalMesh(node,cell)

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

        cell2edge = mesh.cell_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", from_circle_boundary_data)
    def test_from_circle_boundary(self,meshdata ,backend):
        center = bm.from_numpy(meshdata['center'])
        radius = meshdata['radius']
        n = meshdata['n']
        mesh = IntervalMesh.from_circle_boundary(center ,radius , n)
        node = mesh.node
        cell = mesh.cell

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]

        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"])
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

        cell2edge = mesh.cell_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", integrator_data)
    def test_integrator(self,meshdata, backend):
        node = bm.tensor([0,1,2,3],dtype=bm.float64)
        cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int_)

        mesh = IntervalMesh(node,cell)

        q = meshdata["q"]
        qf = mesh.integrator(q)
        bcs , ws = qf.get_quadrature_points_and_weights()

        np.testing.assert_allclose(bm.to_numpy(bcs), meshdata["bcs"] , rtol= 1e-7)
        np.testing.assert_allclose(bm.to_numpy(ws), meshdata["ws"] , rtol= 1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", grad_shape_function_data)
    def test_grad_shape_function(self,meshdata ,backend):
        node = bm.tensor([0,1,2,3],dtype=bm.float64)
        cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int_)

        mesh = IntervalMesh(node,cell)
        q = 2
        qf = mesh.integrator(q)
        bcs , ws = qf.get_quadrature_points_and_weights()
        p = meshdata["p"]
        gphi = mesh.grad_shape_function(bcs,p)
        np.testing.assert_allclose(bm.to_numpy(gphi), meshdata["gphi"] , rtol= 1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", entity_measure_data)
    def test_entity_measure(self,meshdata , backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = IntervalMesh(node,cell)
        cm = mesh.entity_measure(etype="cell")
        nm = mesh.entity_measure(etype="node")
        em = mesh.entity_measure(etype="edge")
        fm = mesh.entity_measure(etype="face")
        np.testing.assert_allclose(bm.to_numpy(cm), meshdata["cm"] , rtol= 1e-7)
        np.testing.assert_allclose(bm.to_numpy(nm), meshdata["nm"] , rtol= 1e-7)
        np.testing.assert_allclose(bm.to_numpy(em), meshdata["em"] , rtol= 1e-7)
        np.testing.assert_allclose(bm.to_numpy(fm), meshdata["fm"] , rtol= 1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", grad_lambda_data)
    def test_grad_lambda(self,meshdata , backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = IntervalMesh(node,cell)
        Dlambda = mesh.grad_lambda()
        print(Dlambda)
        np.testing.assert_allclose(bm.to_numpy(Dlambda), meshdata["Dlambda"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", prolongation_matrix_data)
    def test_prolongation_matrix(self,meshdata , backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        p0 = meshdata['p0']
        p1 = meshdata['p1']
        mesh = IntervalMesh(node,cell)
        PMatrix = mesh.prolongation_matrix(p0,p1).toarray()

        np.testing.assert_allclose(bm.to_numpy(PMatrix), meshdata["prolongation_matrix"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", number_of_local_ipoints_data)
    def test_number_of_local_ipoints(self,meshdata , backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        p = meshdata['p']
        mesh = IntervalMesh(node,cell)
        nlip = mesh.number_of_local_ipoints(p)

        assert nlip == meshdata['nlip']

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", interpolation_points_data)
    def test_interpolation_points(self,meshdata , backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        p = meshdata['p']
        mesh = IntervalMesh(node,cell)
        ipoints = mesh.interpolation_points(p)

        np.testing.assert_allclose(bm.to_numpy(ipoints), meshdata["ipoints"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", cell_normal_data)
    def test_cell_normal(self,meshdata ,backend):
        center = bm.from_numpy(meshdata['center'])
        radius = meshdata['radius']
        n = meshdata['n']
        mesh = IntervalMesh.from_circle_boundary(center ,radius , n)
        cn = mesh.cell_normal()
        np.testing.assert_allclose(bm.to_numpy(cn), meshdata["cn"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", uniform_refine_data)
    def test_uniform_refine(self,meshdata,backend):
        node_init = bm.from_numpy(meshdata['node_init'])
        cell_init = bm.from_numpy(meshdata['cell_init'])
        n = meshdata['n']
        mesh = IntervalMesh(node_init,cell_init)
        mesh.uniform_refine(n)
        
        node = mesh.node
        cell = mesh.cell

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]

        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"])
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

        cell2edge = mesh.cell_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])

        edge2cell = mesh.edge_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(edge2cell), meshdata["edge2cell"])


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", refine_data)
    def test_refine(self,meshdata,backend):
        node_init = bm.from_numpy(meshdata['node_init'])
        cell_init = bm.from_numpy(meshdata['cell_init'])
        isMarkedCell = meshdata['isMarkedCell']
        mesh = IntervalMesh(node_init,cell_init)
        mesh.refine(isMarkedCell)
        
        node = mesh.node
        cell = mesh.cell

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]

        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"])
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

        cell2edge = mesh.cell_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])

        edge2cell = mesh.edge_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(edge2cell), meshdata["edge2cell"])

if __name__ == "__main__":
    pytest.main(["./test_interval_mesh.py","-k", "test_refine"])

        


import pytest
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh.interval_mesh import IntervalMesh
from interval_mesh_data import *


class TestIntervalMeshInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self,meshdata,backend):
        bm.set_backend(backend)
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


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", from_interval_domain_data)
    def test_from_interval_domain(self,meshdata ,backend):
        bm.set_backend(backend)
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

    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", from_mesh_boundary_data)
    def test_from_mesh_boundary(self,meshdata ,backend):
        bm.set_backend(backend)
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


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", from_circle_boundary_data)
    def test_from_circle_boundary(self,meshdata ,backend):
        bm.set_backend(backend)
        center = bm.from_numpy(meshdata['center'])
        radius = meshdata['radius']
        n = meshdata['n']
        mesh = IntervalMesh.from_circle_boundary(center ,radius , n)
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"]
        
        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"],rtol= 1e-7)
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        cell2face = mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face), meshdata["cell2face"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", integrator_data)
    def test_integrator(self,meshdata, backend):
        bm.set_backend(backend)
        node = bm.tensor([0,1,2,3],dtype=bm.float64)
        cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int32)

        mesh = IntervalMesh(node,cell)

        q = meshdata["q"]
        qf = mesh.integrator(q)
        bcs , ws = qf.get_quadrature_points_and_weights()

        np.testing.assert_allclose(bm.to_numpy(bcs), meshdata["bcs"] , rtol= 1e-7)
        np.testing.assert_allclose(bm.to_numpy(ws), meshdata["ws"] , rtol= 1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", grad_shape_function_data)
    def test_grad_shape_function(self,meshdata ,backend):
        bm.set_backend(backend)
        node = bm.tensor([0,1,2,3],dtype=bm.float64)
        cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int32)

        mesh = IntervalMesh(node,cell)
        q = 2
        qf = mesh.integrator(q)
        bcs , ws = qf.get_quadrature_points_and_weights()
        p = meshdata["p"]
        gphi = mesh.grad_shape_function(bcs,p ,variables='x')
        np.testing.assert_allclose(bm.to_numpy(gphi), np.transpose(meshdata["gphi"],[1,0,2,3]) , rtol= 1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", entity_measure_data)
    def test_entity_measure(self,meshdata , backend):
        bm.set_backend(backend)
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
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = IntervalMesh(node,cell)
        Dlambda = mesh.grad_lambda()
        np.testing.assert_allclose(bm.to_numpy(Dlambda), meshdata["Dlambda"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", prolongation_matrix_data)
    def test_prolongation_matrix(self,meshdata , backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        p0 = meshdata['p0']
        p1 = meshdata['p1']
        mesh = IntervalMesh(node,cell)
        PMatrix = mesh.prolongation_matrix(p0,p1).toarray()

        np.testing.assert_allclose(PMatrix, meshdata["prolongation_matrix"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' , 'jax'])
    @pytest.mark.parametrize("meshdata", number_of_local_ipoints_data)
    def test_number_of_local_ipoints(self,meshdata , backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        p = meshdata['p']
        mesh = IntervalMesh(node,cell)
        nlip = mesh.number_of_local_ipoints(p)

        assert nlip == meshdata['nlip']

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", interpolation_points_data)
    def test_interpolation_points(self,meshdata , backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        p = meshdata['p']
        mesh = IntervalMesh(node,cell)
        ipoints = mesh.interpolation_points(p)

        np.testing.assert_allclose(bm.to_numpy(ipoints), meshdata["ipoints"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch' ,'jax'])
    @pytest.mark.parametrize("meshdata", cell_normal_data)
    def test_cell_normal(self,meshdata ,backend):
        bm.set_backend(backend)
        center = bm.from_numpy(meshdata['center'])
        radius = meshdata['radius']
        n = meshdata['n']
        mesh = IntervalMesh.from_circle_boundary(center ,radius , n)
        cn = mesh.cell_normal()
        np.testing.assert_allclose(bm.to_numpy(cn), meshdata["cn"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", uniform_refine_data)
    def test_uniform_refine(self,meshdata,backend):
        bm.set_backend(backend)
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



    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", refine_data)
    def test_refine(self,meshdata,backend):
        bm.set_backend(backend)
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


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", quadrature_formula_data)
    def test_quadrature_formula(self,meshdata,backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        q = meshdata['q']
        qf1 = meshdata['qf1']
        qf2 = meshdata['qf2']
        mesh = IntervalMesh(node,cell)
        qf_test1 = mesh.quadrature_formula(q)[0][0]
        qf_test2 = mesh.quadrature_formula(q)[0][1]

        np.testing.assert_allclose(bm.to_numpy(qf_test1), qf1)
        assert qf_test2 == qf2


        
if __name__ == "__main__":
    #pytest.main(["./test_interval_mesh.py"])
    pytest.main(["./test_interval_mesh.py","-k", "test_uniform_refine"])

        

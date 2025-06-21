import ipdb
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.interval_mesh import IntervalMesh
from fealpy.tests.mesh.interval_mesh_data import * 
from fealpy.mesh import IntervalMesh as IMesh
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TMesh
import matplotlib.pyplot as plt
import numpy as np

# 测试不同的后端
backends = ['numpy', 'torch', 'jax']

class TestIntervalMeshInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param
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
        np.testing.assert_array_equal(bm.to_numpy(cell2edge), meshdata["cell2edge"])


    @pytest.mark.parametrize("meshdata", from_interval_domain_data)
    def test_from_interval_domain(self,meshdata ,backend):
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

    


def test_from_mesh_boundary():
    mesh0 = TMesh.from_box(box = [0,1,0,1], nx = 3 , ny = 3)

    node0 = bm.array(mesh0.entity('node'))
    cell0 = bm.array(mesh0.entity('cell'))
    # 画图
    # 新方法
    mesh1 = TriangleMesh(node0,cell0)
    mesh = IntervalMesh.from_mesh_boundary(mesh1)

    node = mesh.node
    cell = mesh.cell
    mesh2 = IMesh(node,cell)
    fig = plt.figure()
    axis = fig.gca()
    mesh2.add_plot(axis)
    mesh2.find_node(axis,showindex=True)
    mesh2.find_cell(axis,showindex=True)

    # 旧方法
    mesh_old = IMesh.from_mesh_boundary(mesh0)
    fig1 = plt.figure()
    axis1 = fig1.gca()
    mesh_old.add_plot(axis1)
    mesh_old.find_node(axis1,showindex=True)
    mesh_old.find_cell(axis1,showindex=True)
    plt.show()

    assert mesh.node.shape == (12,2)
    assert mesh.cell.shape == (12,2)

    assert mesh.number_of_nodes() == 12
    assert mesh.number_of_cells() == 12

def test_from_circle_boundary():
    center = (0,0)
    radius = 1.0
    n = 10
    # 新方法
    mesh = IntervalMesh.from_circle_boundary(center , radius , n)
    node = mesh.node
    cell = mesh.cell
    mesh1 = IMesh(node,cell)
    fig = plt.figure()
    axis = fig.gca()
    mesh1.add_plot(axis)
    mesh1.find_node(axis,showindex=True)
    mesh1.find_cell(axis,showindex=True)

    #旧方法
    mesh_old = IMesh.from_circle_boundary(center , radius , n)
    fig1 = plt.figure()
    axis1 = fig1.gca()
    mesh_old.add_plot(axis1)
    mesh_old.find_node(axis1,showindex=True)
    mesh_old.find_cell(axis1,showindex=True)
    plt.show()

    assert mesh.node.shape == (10,2)
    assert mesh.cell.shape == (10,2)

    assert mesh.number_of_nodes() == 10
    assert mesh.number_of_cells() == 10

def test_integrator():
    node = bm.tensor([0,1,2,3],dtype=bm.float64)
    cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int_)
    q = 2

    mesh_old = IMesh(node,cell)
    integrator_old = mesh_old.integrator(q)
    bcs0 , ws0 = integrator_old.get_quadrature_points_and_weights()

    mesh = IntervalMesh(node, cell)
    integrator = mesh.integrator(q)
    bcs ,ws = integrator.get_quadrature_points_and_weights()

    assert bcs.all() == bcs0.all()
    assert ws.all() == ws0.all()

def test_grad_shape_function():
    node = bm.tensor([0,1,2,3],dtype=bm.float64)
    cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int_)
    mesh = IntervalMesh(node, cell)
    qf = mesh.integrator(2)
    bcs,ws = qf.get_quadrature_points_and_weights()
    p = 2
    gphi = mesh.grad_shape_function(bcs,p)

def test_entity_measure():
    node = bm.tensor([0,1,2,3],dtype=bm.float64)
    cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int_)
    mesh = IntervalMesh(node, cell)
    cm = mesh.entity_measure(etype='cell' )
    mesh_old = IMesh(node,cell)
    cm_old = mesh_old.entity_measure('cell')

    assert cm.all() == cm_old.all()

def test_grad_lambda():
    node = bm.tensor([0,1,2,3],dtype=bm.float64)
    cell = bm.tensor([[0,1],[1,2],[2,3]],dtype=bm.int_)
    mesh = IntervalMesh(node, cell)
    Dlambda = mesh.grad_lambda()
    mesh_old = IMesh(node,cell)
    Dlambda_old = mesh_old.grad_lambda()

    assert Dlambda.all() == Dlambda_old.all()

def test_prolongation_matrix():
    Interval = [0,1]
    n = 5
    mesh = IntervalMesh.from_interval_domain(Interval ,n)
    mesh_old = IMesh.from_interval_domain(Interval,n)
    p0 = 1
    p1 = 2
    P = mesh.prolongation_matrix(p0,p1)
    #P_old = mesh_old.prolongation_matrix(p0,p1)
    #cell2ipoint NotImplementedError

def test_number_of_local_ipoints():
    p = 4
    Interval = [0,1]
    n = 5
    mesh = IntervalMesh.from_interval_domain(Interval ,n)
    numoflipoint = mesh.number_of_local_ipoints(p)
    assert numoflipoint == 5

def test_interpolation_points():
    Interval = [0,1]
    n = 5
    mesh = IntervalMesh.from_interval_domain(Interval ,n)
    mesh_old = IMesh.from_interval_domain(Interval ,n)
    node = mesh.node
    p0 = 1
    p1 = 4
    ipoint0 = mesh.interpolation_points(p0)
    ipoint1 = mesh.interpolation_points(p1)
    ipoint_old = mesh_old.interpolation_points(p1)

    assert ipoint0.all() == node.all()
    assert ipoint1.all() == ipoint_old.all()

def test_cell_normal():
    center = (0,0)
    radius = 1.0
    n = 10
    mesh = IntervalMesh.from_circle_boundary(center , radius , n)
    cn = mesh.cell_normal()
    mesh_old = IMesh.from_circle_boundary(center , radius , n)
    cn_old = mesh_old.cell_normal()

    assert np.array_equal(cn , cn_old)

def test_uniform_refine():
    n = 1
    Interval = [0,1]
    nx = 1
    mesh = IntervalMesh.from_interval_domain(Interval ,nx)
   
    mesh.uniform_refine(n)
   
    node = mesh.node
    
    cell = mesh.cell
    mesh0 = IMesh(mesh.node,mesh.cell)
    fig = plt.figure()
    axis = fig.gca()
    mesh0.add_plot(axis)
    mesh0.find_node(axis,showindex=True)
    mesh0.find_cell(axis,showindex=True)
    plt.show()

def test_refine():
    Interval = [0,1]
    n = 2
    mesh = IntervalMesh.from_interval_domain(Interval ,n)
    isMarkedCell = bm.tensor([True,False])
    mesh.refine(isMarkedCell)

    node = mesh.node
    cell = mesh.cell
    mesh0 = IMesh(node,cell)
    fig = plt.figure()
    axis = fig.gca()
    mesh0.add_plot(axis)
    mesh0.find_node(axis,showindex=True)
    mesh0.find_cell(axis,showindex=True)
    plt.show()

    mesh_old = IMesh.from_interval_domain(Interval,n)
    mesh_old.refine(isMarkedCell)
    node_old = mesh_old.entity('node')
    cell_old = mesh_old.entity('cell')

    assert np.array_equal(node,node_old)
    assert np.array_equal(cell,cell_old)

TIM = TestIntervalMeshInterfaces()

if __name__ == "__main__":
    pytest.main(["./test_interval_mesh.py"])
'''
if __name__ == "__main__":
    #test_interval_mesh_init()
    #test_from_interval_domain()
    # test_from_mesh_boundary()
    test_from_circle_boundary()
    # test_integrator()
    # test_grad_shape_function()
    # test_entity_measure()
    # test_grad_lambda()
    # test_prolongation_matrix()         
    # test_number_of_local_ipoints()
    # test_interpolation_points()
    # test_cell_normal()               
    # test_uniform_refine()              
    # test_refine()
'''
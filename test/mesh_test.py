"""
"""


import pytest

import numpy as np
from scipy.sparse.linalg import spsolve


def test_triangle_mesh():
    from fealpy.mesh import TriangleMesh
    node = np.array([
        (0.0, 0.0), # 0 号点
        (1.0, 0.0), # 1 号点
        (1.0, 1.0), # 2 号点
        (0.0, 1.0), # 3 号点
        ], dtype=np.float64)
    cell = np.array([
        (1, 2, 0), # 0 号单元
        (3, 0, 2), # 1 号单元
        ], dtype=np.int_)

    mesh = TriangleMesh(node, cell)

    # 获取测试
    assert id(node) == id(mesh.entity('node'))
    assert id(cell) == id(mesh.entity('cell'))

    edge = np.array([
        [0, 1],
        [2, 0],
        [3, 0],
        [1, 2],
        [2, 3],
        ], dtype=np.int_)
    assert np.all(edge == mesh.entity('edge'))

    isBdNode = mesh.ds.boundary_node_flag()
    assert np.all(isBdNode)

    isBdEdge = mesh.ds.boundary_edge_flag()
    assert isBdEdge.sum() == 4
    assert isBdEdge[1] == False

    cell2node = mesh.ds.cell_to_node()
    cell2edge = mesh.ds.cell_to_edge()
    cell2cell = mesh.ds.cell_to_cell()

    edge2node = mesh.ds.edge_to_node()
    edge2edge = mesh.ds.edge_to_edge()
    edge2cell = mesh.ds.edge_to_cell()

    node2node = mesh.ds.node_to_node()
    node2edge = mesh.ds.node_to_edge()
    node2cell = mesh.ds.node_to_cell()

    # 加密测试 
    mesh.uniform_refine()

def test_tetrahedron_mesh():
    from fealpy.mesh import TetrahedronMesh

    node = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
    mesh = TetrahedronMesh(node, cell)

    # 获取测试
    assert id(node) == id(mesh.entity('node'))
    assert id(cell) == id(mesh.entity('cell'))


    isBdNode = mesh.ds.boundary_node_flag()
    assert np.all(isBdNode)

    isBdEdge = mesh.ds.boundary_edge_flag()
    assert np.all(isBdEdge) 

    isBdFace = mesh.ds.boundary_face_flag()
    assert np.all(isBdFace)

    cell2node = mesh.ds.cell_to_node()
    cell2edge = mesh.ds.cell_to_edge()
    cell2face = mesh.ds.cell_to_face()
    cell2cell = mesh.ds.cell_to_cell()

    edge2node = mesh.ds.edge_to_node()
    edge2edge = mesh.ds.edge_to_edge()
    edge2face = mesh.ds.edge_to_face()
    edge2cell = mesh.ds.edge_to_cell()

    face2node = mesh.ds.face_to_node()
    face2edge = mesh.ds.face_to_edge()
    face2face = mesh.ds.face_to_face()
    face2cell = mesh.ds.face_to_cell()

    node2node = mesh.ds.node_to_node()
    node2edge = mesh.ds.node_to_edge()
    node2face = mesh.ds.node_to_face()
    node2cell = mesh.ds.node_to_cell()
    
    mesh.uniform_refine()


def test_quadrangle_mesh():
    from fealpy.mesh import QuadrangleMesh
    node = np.array([
        (0.0, 0.0), # 0 号点
        (0.0, 1.0), # 1 号点
        (0.0, 2.0), # 2 号点
        (1.0, 0.0), # 3 号点
        (1.0, 1.0), # 4 号点
        (1.0, 2.0), # 5 号点
        (2.0, 0.0), # 6 号点
        (2.0, 1.0), # 7 号点
        (2.0, 2.0), # 8 号点
        ], dtype=np.float64)
    cell = np.array([
        (0, 3, 4, 1), # 0 号单元
        (1, 4, 5, 3), # 1 号单元
        (3, 6, 7, 4), # 2 号单元
        (4, 7, 8, 5), # 3 号单元
        ], dtype=np.int_)

    mesh = QuadrangleMesh(node, cell)

    cell2node = mesh.ds.cell_to_node()
    cell2edge = mesh.ds.cell_to_edge()
    cell2cell = mesh.ds.cell_to_cell()

    edge2node = mesh.ds.edge_to_node()
    edge2edge = mesh.ds.edge_to_edge()
    edge2cell = mesh.ds.edge_to_cell()

    node2node = mesh.ds.node_to_node()
    node2edge = mesh.ds.node_to_edge()
    node2cell = mesh.ds.node_to_cell()

    # 加密测试 
    mesh.uniform_refine()

def test_hexahedron_mesh():
    from fealpy.mesh import HexahedronMesh
    node = np.array([
        (0.0, 0.0, 0.0), # 0 号点
        (0.0, 1.0, 0.0), # 1 号点
        (1.0, 1.0, 0.0), # 2 号点
        (1.0, 0.0, 0.0), # 3 号点
        (0.0, 0.0, 1.0), # 4 号点
        (0.0, 1.0, 1.0), # 5 号点
        (1.0, 1.0, 1.0), # 6 号点
        (1.0, 0.0, 1.0), # 7 号点
        ], dtype=np.float64)
    cell = np.array([
        (0, 1, 2, 3, 4, 5, 6, 7), # 0 号单元
        ], dtype=np.int_)

    mesh = HexahedronMesh(node, cell)
    
    #mesh.uniform_refine()

    cell2node = mesh.ds.cell_to_node()
    cell2edge = mesh.ds.cell_to_edge()
    cell2face = mesh.ds.cell_to_face()
    cell2cell = mesh.ds.cell_to_cell()

    edge2node = mesh.ds.edge_to_node()
    edge2edge = mesh.ds.edge_to_edge()
    edge2face = mesh.ds.edge_to_face()
    edge2cell = mesh.ds.edge_to_cell()

    face2node = mesh.ds.face_to_node()
    face2edge = mesh.ds.face_to_edge()
    face2face = mesh.ds.face_to_face()
    face2cell = mesh.ds.face_to_cell()

    node2node = mesh.ds.node_to_node()
    node2edge = mesh.ds.node_to_edge()
    node2face = mesh.ds.node_to_face()
    node2cell = mesh.ds.node_to_cell()

@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_lagrange_triangle_mesh(p):
    from fealpy.mesh import LagrangeTriangleMesh
    node = np.array([
        (0.0, 0.0), # 0 号点
        (1.0, 0.0), # 1 号点
        (1.0, 1.0), # 2 号点
        (0.0, 1.0), # 3 号点
        ], dtype=np.float64)
    cell = np.array([
        (1, 2, 0), # 0 号单元
        (3, 0, 2), # 1 号单元
        ], dtype=np.int_)

    mesh = LagrangeTriangleMesh(node, cell, p=p)

    cell2node = mesh.ds.cell_to_node()
    cell2edge = mesh.ds.cell_to_edge()
    cell2cell = mesh.ds.cell_to_cell()

    edge2node = mesh.ds.edge_to_node()
    edge2edge = mesh.ds.edge_to_edge()
    edge2cell = mesh.ds.edge_to_cell()

    node2node = mesh.ds.node_to_node()
    node2edge = mesh.ds.node_to_edge()
    node2cell = mesh.ds.node_to_cell()

@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_lagrange_quadrangle_mesh(p):
    from fealpy.mesh import LagrangeQuadrangleMesh
    node = np.array([
        (0.0, 0.0), # 0 号点
        (0.0, 1.0), # 1 号点
        (0.0, 2.0), # 2 号点
        (1.0, 0.0), # 3 号点
        (1.0, 1.0), # 4 号点
        (1.0, 2.0), # 5 号点
        (2.0, 0.0), # 6 号点
        (2.0, 1.0), # 7 号点
        (2.0, 2.0), # 8 号点
        ], dtype=np.float64)
    cell = np.array([
        (0, 1, 3, 4), # 0 号单元
        (1, 3, 4, 5), # 1 号单元
        (3, 4, 6, 7), # 2 号单元
        (4, 5, 7, 8), # 3 号单元
        ], dtype=np.int_)

    mesh = LagrangeQuadrangleMesh(node, cell, p=p)

    cell2node = mesh.ds.cell_to_node()
    cell2edge = mesh.ds.cell_to_edge()
    cell2cell = mesh.ds.cell_to_cell()

    edge2node = mesh.ds.edge_to_node()
    edge2edge = mesh.ds.edge_to_edge()
    edge2cell = mesh.ds.edge_to_cell()

    node2node = mesh.ds.node_to_node()
    node2edge = mesh.ds.node_to_edge()
    node2cell = mesh.ds.node_to_cell()

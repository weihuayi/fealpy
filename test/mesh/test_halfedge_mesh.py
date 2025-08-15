import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d
from fealpy.mesh import TriangleMesh, QuadrangleMesh, PolygonMesh
from halfedge_mesh_data import *
import pytest

class TestHalfEdgeMesh():
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        mesh = HalfEdgeMesh2d(data["node"], bm.array(data['halfedge']))
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NHE = mesh.number_of_halfedges()
        NBE = mesh.number_of_boundary_edges()
        NBN = mesh.number_of_boundary_nodes()
        NV = mesh.number_of_vertices_of_cells()
        NEC = mesh.number_of_edges_of_cells()
        NNC = mesh.number_of_nodes_of_cells()
        NFC = mesh.number_of_faces_of_cells()
        hcell = mesh.hcell
        hedge = mesh.hedge
        hnode = mesh.hnode
        TD = mesh.TD
        assert NN == data["NN"]
        assert NE == data["NE"]
        assert NC == data["NC"]
        assert NHE == data["NHE"]
        assert NBE == data["NBE"]
        assert NBN == data["NBN"]
        assert TD == data["TD"]
        np.testing.assert_array_equal(hcell, data["hcell"])
        np.testing.assert_array_equal(hedge, data["hedge"])
        np.testing.assert_array_equal(hnode, data["hnode"])
        np.testing.assert_array_equal(NV, data["NV"])
        np.testing.assert_array_equal(NEC, data["NEC"])
        np.testing.assert_array_equal(NNC, data["NNC"])
        np.testing.assert_array_equal(NFC, data["NFC"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", from_mesh)
    def test_from_mesh(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        halfedge = mesh.halfedge
        node = mesh.node
        np.testing.assert_array_equal(data['halfedge'], halfedge)
        np.testing.assert_array_equal(data['node'], node)
 
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", cell_to_node_edge_cell)
    def test_cell_to_node_edge_cell(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'quad':
            tmesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        bool_cell2node = mesh.cell_to_node(return_sparse=True)
        bool_cell2edge = mesh.cell_to_edge(return_sparse=True)
        bool_cell2cell = mesh.cell_to_cell(return_sparse=True)
        cell2node = mesh.cell_to_node(return_sparse=False)
        cell2edge = mesh.cell_to_edge(return_sparse=False)
        cell2cell = mesh.cell_to_cell(return_sparse=False)
        cell2halfedge = mesh.cell_to_halfedge()
        np.testing.assert_array_equal(data['bool_cell2node'], bool_cell2node.toarray())
        np.testing.assert_array_equal(data['bool_cell2edge'], bool_cell2edge.toarray())
        np.testing.assert_array_equal(data['bool_cell2cell'], bool_cell2cell.toarray())
        for arr1, arr2 in zip(cell2node,data['cell2node']):
            np.testing.assert_array_equal(arr1, arr2)
        for arr1, arr2 in zip(cell2edge,data['cell2edge']):
            np.testing.assert_array_equal(arr1, arr2)
        for arr1, arr2 in zip(cell2cell,data['cell2cell']):
            np.testing.assert_array_equal(arr1, arr2)
        for arr1, arr2 in zip(cell2halfedge,data['cell2halfedge']):
            np.testing.assert_array_equal(arr1, arr2)



    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", edge_to_node_edge_cell)
    def test_edge_to_node_edge_cell(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'quad':
            tmesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        bool_edge2node = mesh.edge_to_node(return_sparse=True)
        edge2node = mesh.edge_to_node(return_sparse=False)
        edge2edge = mesh.edge_to_edge() # bool
        edge2cell = mesh.edge_to_cell()
        np.testing.assert_array_equal(data['bool_edge2node'], bool_edge2node.toarray())
        np.testing.assert_array_equal(data['edge2edge'], edge2edge.toarray())
        for arr1, arr2 in zip(edge2node,data['edge2node']):
            np.testing.assert_array_equal(arr1, arr2)
        for arr1, arr2 in zip(edge2cell,data['edge2cell']):
            np.testing.assert_array_equal(arr1, arr2)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", node_to_node_edge_cell)
    def test_node_to_node_edge_cell(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'quad':
            tmesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        node2node = mesh.node_to_node() #bool
        node2cell = mesh.node_to_cell()
        np.testing.assert_array_equal(data['node2node'], node2node.toarray())
        np.testing.assert_array_equal(data['node2cell'], node2cell.toarray())


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", halfedge_to)
    def test_halfedge_to(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'quad':
            tmesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        halfedge2cellnum = mesh.halfedge_to_cell_location_number()
        halfedge2edge = mesh.halfedge_to_edge()
        np.testing.assert_array_equal(data['halfedge2cellnum'], halfedge2cellnum)
        np.testing.assert_array_equal(data['halfedge2edge'], halfedge2edge)
 
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", ipoint)
    def test_ipoint(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'quad':
            tmesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        e2p = mesh.edge_to_ipoint(p=3)
        c2p = mesh.cell_to_ipoint(p=3)
        for arr1,arr2 in zip(e2p, data['e2p']):
            np.testing.assert_array_equal(arr1, arr2)
        for arr1,arr2 in zip(c2p, data['c2p']):
            np.testing.assert_array_equal(arr1, arr2)
    
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", boundary)
    def test_boundary(self, data, backend):
        bm.set_backend(backend)
        if data['mesh'] =='tri':
            tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'quad':
            tmesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        elif data['mesh'] == 'poly':
            node = np.array([[0.33333333, 0.33333333],
       [0.83333333, 0.33333333],
       [0.16666667, 0.66666667],
       [0.66666667, 0.66666667],
       [0.        , 0.5       ],
       [0.25      , 0.        ],
       [0.25      , 1.        ],
       [0.75      , 0.        ],
       [0.75      , 1.        ],
       [1.        , 0.5       ],
       [0.        , 0.        ],
       [0.        , 1.        ],
       [0.5       , 0.        ],
       [0.5       , 1.        ],
       [1.        , 0.        ],
       [1.        , 1.        ]])
            cell = np.array([ 0,  2,  4, 10,  5,  2,  6, 11,  4,  1,  3,  0,  5,
                             12,  7,  2,  0, 3,  8, 13,  6,  1,  7, 14,  9,  3,
                             1,  9, 15,  8], dtype=np.int32) 
            cellLocation = np.array([ 0,  5,  9, 15, 21, 25, 30],
                                    dtype=np.int32)
            cell = (bm.array(cell), bm.array(cellLocation))
            tmesh = PolygonMesh(bm.array(node), cell)
        mesh = HalfEdgeMesh2d.from_mesh(tmesh)
        nex = mesh.nex_boundary_halfedge()
        isBdNode = mesh.boundary_node_flag()
        isBdEdge = mesh.boundary_edge_flag()
        isBdCell = mesh.boundary_cell_flag()
        isBdHEdge = mesh.boundary_halfedge_flag()
        np.testing.assert_array_equal(data['isBdNode'], isBdNode)
        np.testing.assert_array_equal(data['isBdEdge'], isBdEdge)
        np.testing.assert_array_equal(data['isBdCell'], isBdCell)
        np.testing.assert_array_equal(data['isBdHalfedge'], isBdHEdge)


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", convexity)
    def test_convexity(self, data, backend):
        bm.set_backend(backend)
        pmesh = PolygonMesh.distorted_concave_rhombic_quadrilaterals_mesh(nx=1,ny=2)
        mesh = HalfEdgeMesh2d.from_mesh(pmesh)
        mesh.convexity()
        node = mesh.node
        halfedge = mesh.halfedge
        clevel = mesh.celldata['level']
        hlevel = mesh.halfedgedata['level']
        np.testing.assert_array_equal(data['halfedge'],halfedge)
        np.testing.assert_allclose(data['node'],node, atol=1e-12)
        np.testing.assert_array_equal(data['clevel'],clevel)
        np.testing.assert_array_equal(data['hlevel'],hlevel)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", grad_shape_function)
    def test_grad_shape_function(self, data, backend):
        bm.set_backend(backend)
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=1)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        bc = mesh.multi_index_matrix(3,2)/3
        grad_lambda = mesh.grad_lambda()
        print(grad_lambda)
        grad_shape = mesh.grad_shape_function(bc,p=2)
        grad_shape = grad_shape.transpose(1,0,2)
        np.testing.assert_allclose(data['grad_shape_x'], grad_shape, atol=1e-12)
        grad_shape_u = mesh.grad_shape_function(bc,p=2,variables='u')
        grad_shape_u = grad_shape_u.transpose(1,0,2)
        np.testing.assert_allclose(data['grad_shape_u'], grad_shape_u, atol=1e-12)

    def test_from_bdf_and_to_vtk(self):
        file_path = 'Sheet_Metal_20250717_Before_Opt_v2.bdf'

        half_edge_mesh = HalfEdgeMesh2d.from_bdf(file_path)
        half_edge_mesh.to_vtk(fname="halfedge_mesh.vtu")

 
if __name__ == "__main__":
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_init"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_from_mesh"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_cell_to_node_edge_cell"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_edge_to_node_edge_cell"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_node_to_node_edge_cell"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_halfedge_to"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_ipoint"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_boundary"])
    pytest.main(["-v", "test_halfedge_mesh.py","-k","test_convexity"])
    a = TestHalfEdgeMesh() 
    #a.test_init(init_data[0], "numpy")
    #a.test_from_mesh(from_mesh[1], "numpy")
    #a.test_cell_to_node_edge_cell(cell_to_node_edge_cell[2], "pytorch")
    #a.test_cell_to_node(cell_to_node[0], "pytorch")
    #a.test_edge_to_node_edge_cell(edge_to_node_edge_cell[0], "pytorch")
    #a.test_node_to_node_edge_cell(edge_to_node_edge_cell[1], "pytorch")
    #a.test_halfedge_to(edge_to_node_edge_cell[2], "pytorch")
    #a.test_ipoint(ipoint[2], "numpy")
    #a.test_convexity(convexity[0], "pytorch")
    a.test_grad_shape_function(grad_shape_function[0], "pytorch")



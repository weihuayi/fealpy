import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.backend import backend_manager as bm

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d

from scaled_monomial_space_2d_data import *

class TestScaledMonomialSpace2d():
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",multi_index_matrix)
    def test_multi_index_matrix(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=5)
        multi_index_matrix = space.multi_index_matrix(p=4)
        np.testing.assert_equal(multi_index_matrix, data["multi_index_matrix"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",cell_to_dof)
    def test_cell_to_dof(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=5)
        cell2dof = space.cell_to_dof(p=4)
        np.testing.assert_equal(cell2dof, data["cell2dof"])
        ldof = space.number_of_local_dofs(p=4)
        gdof = space.number_of_global_dofs(p=4)
        assert ldof == data["ldof"]
        assert gdof == data["gdof"]

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",index)
    def test_index(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=3)
        diff_index_1 = space.diff_index_1() 
        diff_index_2 = space.diff_index_2()
        edge_index_1 = space.edge_index_1()
        face_index_1 = space.edge_index_1()
        a = diff_index_1.values()
        for x,y in zip(a, data["index_1"].values()):
            np.testing.assert_equal(x, y)
        a = diff_index_2.values()
        for x,y in zip(a, data["index_2"].values()):
            np.testing.assert_equal(x, y)
        a = edge_index_1.values()
        for x,y in zip(a, data["edge_index_1"].values()):
            np.testing.assert_equal(x, y)
        a = face_index_1.values()
        for x,y in zip(a, data["face_index_1"].values()):
            np.testing.assert_equal(x, y)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",index)
    def test_basis(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=3)
        edge_basis = space.edge_basis(point, index=np.array([0,1]))
        print(edge_index)
        edge_basis_with_bcs = space.edge_basis_with_barycentric(point)
        


      
 
if __name__ == '__main__':
    ts = TestScaledMonomialSpace2d()
    #ts.test_multi_index_matrix('numpy', multi_index_matrix[0])
    #ts.test_multi_index_matrix('pytorch', multi_index_matrix[0])
    #ts.test_cell_to_dof('pytorch', cell_to_dof[0])
    ts.test_basis('pytorch', index[0])
























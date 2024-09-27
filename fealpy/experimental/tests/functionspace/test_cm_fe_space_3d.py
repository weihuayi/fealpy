import pytest
import numpy as np
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.experimental.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d
from fealpy.experimental.tests.functionspace.cm_fe_space_3d_data import *
import ipdb

class TestCmfespace3d:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", get_dof_index )
    def test_get_dof_index(self, data, backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        get_dof_index = space.dof_index["all"]        
        np.testing.assert_array_equal(get_dof_index, data["dof_index"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", number_of_internal_global_dofs)
    def test_number_of_internal_dofs(self, data, backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        gdof = space.number_of_global_dofs()
        idof = space.number_of_internal_dofs(data["etype"])
        #np.testing.assert_array_equal(ldof, data["ldof"])
        assert idof == data["ldof"]
        assert gdof == data["gdof"]


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", nefc_to_internal_dof)
    def test_nefc_to_internal_dofs(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        n2d = space.node_to_dof()
        e2id = space.edge_to_internal_dof()
        f2id = space.face_to_internal_dof()
        c2id = space.cell_to_internal_dof()
        np.testing.assert_array_equal(n2d, data["node"])
        np.testing.assert_array_equal(e2id, data["edge"])
        np.testing.assert_array_equal(f2id, data["face"])
        np.testing.assert_array_equal(c2id, data["cell"])


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", nefc_to_internal_dof)
    def test_cell_to_dofs(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],3,3,3)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        c2d = space.cell_to_dof()
 





if __name__=="__main__":

    t = TestCmfespace3d()
    # print(1)
    #t.test_get_dof_index(get_dof_index[0], "numpy")
    # t.test_get_dof_index(get_dof_index[0], "pytorch")
    # t.test_number_of_internal_dofs(number_of_global_dofs, "pytorch")
    t.test_cell_to_dofs(nefc_to_internal_dof[0], "numpy")

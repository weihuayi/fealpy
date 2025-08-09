import pytest
from fealpy.backend import backend_manager as bm
import importlib

from mesh_base_data import *

class TestMeshBase:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", sub_mesh_class1)
    def test_get_bd_mesh1(self, backend, data):
        class_name = data["class_name"]
        sub_class_name = data["sub_class_name"]
        mesh = getattr(importlib.import_module("fealpy.mesh"), class_name).from_box()

        bd_node, bd_cell, bd_node_idx, bd_face_idx = mesh.get_boundary_mesh()
        sub_cls = getattr(importlib.import_module("fealpy.mesh"), sub_class_name)
        bd_mesh = sub_cls(bd_node, bd_cell)
        # bd_mesh.to_vtk(fname=f"bd_mesh_{class_name}.vtu")
        np.testing.assert_allclose(bd_mesh.node, mesh.node[bd_node_idx], atol=1e-7)
        np.testing.assert_array_equal(bd_node_idx[bd_mesh.cell], mesh.face[bd_face_idx])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", sub_mesh_class2)
    def test_get_bd_mesh2(self, backend, data):
        class_name = data["class_name"]
        sub_class_name = data["sub_class_name"]
        mesh = getattr(importlib.import_module("fealpy.mesh"), class_name)()

        bd_node, bd_cell, bd_node_idx, bd_face_idx = mesh.get_boundary_mesh()
        sub_cls = getattr(importlib.import_module("fealpy.mesh"), sub_class_name)
        bd_mesh = sub_cls(bd_node, bd_cell)
        # bd_mesh.to_vtk(fname=f"bd_mesh_{class_name}.vtu")
        np.testing.assert_allclose(bd_mesh.node, mesh.node[bd_node_idx], atol=1e-7)


if __name__ == "__main__":
    pytest.main(["./test_mesh_base.py", "-k", "test_get_bd_mesh"])
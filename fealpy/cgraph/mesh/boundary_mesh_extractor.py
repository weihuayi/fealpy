from typing import Type
import importlib

from ..nodetype import CNodeType, PortConf, DataType


sub_mesh_class_map = {
    "tri": "interval",
    "tet": "triangle",
    "quad": "interval",
    "hex": "quadrangle",
}

def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)


class BoundaryMeshExtractor(CNodeType):
    r"""Get the boundary mesh from input mesh.

    Inputs:
        input_mesh (MeshType): The input mesh object.

    Outputs:
            boundary_mesh (MeshType): The boundary mesh of the input mesh.
            node_idx (array): the index of nodes in the boundary mesh
                within nodes of original mesh.
            face_idx (array): the index of cells in the boundary mesh
                within faces of original mesh.
    """
    TITLE: str = "提取边界网格"
    PATH: str = "网格.操作"
    INPUT_SLOTS = [
        PortConf("input_mesh", DataType.MESH, title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("boundary_mesh", DataType.MESH, title="边界网格"),
        PortConf("node_idx", DataType.TENSOR, title="点编号"),
        PortConf("face_idx", DataType.TENSOR, title="面编号")
    ]

    @staticmethod
    def run(input_mesh):
        origin_mesh_type = input_mesh.meshtype
        sub_mesh_type = sub_mesh_class_map.get(origin_mesh_type, None)
        bd_node, bd_cell, node_idx, face_idx = input_mesh.get_boundary_mesh()
        MeshClass = get_mesh_class(sub_mesh_type)
        boundary_mesh = MeshClass(bd_node, bd_cell)

        return boundary_mesh, node_idx, face_idx





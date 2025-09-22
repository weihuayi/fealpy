from typing import Type
import importlib

from ..nodetype import CNodeType, PortConf, DataType
from ...mesh import BdfFileParser, NodeSection, ElementSection


mesh_type_map = {
    'CTRIA3': 'triangle',
    'CQUAD4': 'quadrangle',
    'CTETRA': 'tetrahedron',
    'CHEXA': 'hexahedron'
}

def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)


class BdfMeshReader(CNodeType):
    r"""Create a mesh from Nastran *.bdf file.

    Inputs:
        input_bdf_file (str): Path to the Nastran *.bdf file.

    Outputs:
        output_mesh (MeshType): The mesh object created.
    """
    TITLE: str = "BDF 网格读取"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("input_bdf_file", DataType.STRING)
    ]
    OUTPUT_SLOTS = [
        PortConf("output_mesh", DataType.MESH)
    ]

    @staticmethod
    def run(input_bdf_file):
        bdf_parser = BdfFileParser().parse(input_bdf_file)
        node_section = bdf_parser.get_section(NodeSection)
        cell_section = bdf_parser.get_section(ElementSection)
        node = node_section.node
        node_map = node_section.node_map
        cell = None
        mesh_type = None
        for key, value in cell_section.cell.items():
            mesh_type = mesh_type_map.get(key)
            cell = node_map[value]

        MeshClass = get_mesh_class(mesh_type)
        return MeshClass(node, cell)



from typing import Type
import importlib

from ..nodetype import CNodeType, PortConf, DataType


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
        PortConf("input_bdf_file", DataType.STRING, title="文件路径", default="")
    ]
    OUTPUT_SLOTS = [
        PortConf("output_mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(input_bdf_file):
        from fealpy.backend import bm
        from fealpy.mesh import BdfFileParser, NodeSection, ElementSection

        bdf_parser = BdfFileParser().parse(input_bdf_file)
        node_section = bdf_parser.get_section(NodeSection)
        cell_section = bdf_parser.get_section(ElementSection)
        node = node_section.node
        node_map = node_section.node_map
        cell = None
        mesh_type = None

        for key, value in cell_section.cell.items():
            next_mesh_type = mesh_type_map.get(key)

            if mesh_type is None:
                mesh_type = next_mesh_type
                cell = node_map[value]

            elif mesh_type == next_mesh_type:
                cell = bm.concat([cell, node_map[value]], axis=0)

            elif (mesh_type, next_mesh_type) == ('triangle', 'quadrangle'):
                mesh_type = 'quadrangle'
                cell = BdfMeshReader.fill_minus_one(cell, -1, 4)
                cell = bm.concat([cell, node_map[value]], axis=0)

            elif (mesh_type, next_mesh_type) == ('quadrangle', 'triangle'):
                mesh_type = 'quadrangle'
                new_cell = BdfMeshReader.fill_minus_one(node_map[value], -1, 4)
                cell = bm.concat([cell, new_cell], axis=0)

            elif (mesh_type, next_mesh_type) == ('tetrahedron', 'hexahedron'):
                mesh_type = 'hexahedron'
                cell = BdfMeshReader.fill_minus_one(cell, -1, 8)
                cell = bm.concat([cell, node_map[value]], axis=0)

            elif (mesh_type, next_mesh_type) == ('hexahedron', 'tetrahedron'):
                mesh_type = 'hexahedron'
                new_cell = BdfMeshReader.fill_minus_one(node_map[value], -1, 8)
                cell = bm.concat([cell, new_cell], axis=0)

            else:
                raise ValueError(f'{mesh_type} -> {next_mesh_type} is not supported')

        # for key, value in cell_section.cell.items():
        #     mesh_type = mesh_type_map.get(key)
        #     cell = node_map[value]

        MeshClass = get_mesh_class(mesh_type)
        return MeshClass(node, cell)

    @staticmethod
    def fill_minus_one(array, axis: int, length: int):
        from fealpy.backend import bm
        minus_one_shape = list(array.shape)
        current_length = array.shape[axis]
        minus_one_shape[axis] = length - current_length
        minus_one = bm.full(minus_one_shape, -1, **bm.context(array))
        return bm.concat([array, minus_one], axis=axis)

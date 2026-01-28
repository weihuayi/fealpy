from typing import Type

from ..nodetype import CNodeType, PortConf, DataType


class InpMeshReader(CNodeType):
    r"""Create a mesh from *.inp file.

    Inputs:
        input_inp_file (str): Path to the *.inp file.

    Outputs:
        output_mesh (MeshType): The mesh object created.
    """
    TITLE: str = "INP 网格读取"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("input_inp_file", DataType.STRING, title="文件路径", default="")
    ]
    OUTPUT_SLOTS = [
        PortConf("output_mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(input_inp_file):
        from fealpy.backend import bm
        from fealpy.mesh import (
            MeshData,
            TetrahedronMesh,
            InpFileParser,
        )
        inp_parser = InpFileParser()
        inp_parser.parse(input_inp_file)
        output_mesh = inp_parser.to_mesh(TetrahedronMesh, MeshData)

        return output_mesh
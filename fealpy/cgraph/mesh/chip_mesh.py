from typing import Type
import importlib

from ..nodetype import CNodeType, PortConf, DataType


class ChipMesh2D(CNodeType):
    
    TITLE: str = "Chip Mesh 2D"
    PATH: str = "mesh.chipmesh"
    INPUT_SLOTS = [
        PortConf("box", DataType.NONE, 0, default = [0.0, 1.0, 0.0, 1.0]),
        PortConf("holes", DataType.NONE, 0, default = [[0.3, 0.3, 0.1], [0.3, 0.7, 0.1], [0.7, 0.3, 0.1], [0.7, 0.7, 0.1]]),
        PortConf("h", DataType.FLOAT, default=0.02, min_val=0.001)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH)
    ]

    @staticmethod
    def run(box, holes, h):
        from fealpy.mesh import  TriangleMesh
        return TriangleMesh.from_box_with_circular_holes(box = box, holes = holes, h = h)

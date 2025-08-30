from typing import Type
import importlib

from ..nodetype import CNodeType, PortConf, DataType


def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)


class Box2d(CNodeType):
    r"""Create a mesh in a box-shaped 2D area.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float], optional): Domain.
        nx (int, optional): Segments on x direction.
        ny (int, optional): Segments on y direction.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "Box 2D"
    PATH: str = "mesh.creation"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, default="triangle", items=["triangle", "quadrangle"]),
        PortConf("domain", DataType.NONE),
        PortConf("nx", DataType.INT, default=10, min_val=1),
        PortConf("ny", DataType.INT, default=10, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH)
    ]

    @staticmethod
    def run(mesh_type, domain, nx, ny):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"nx": nx, "ny": ny}
        if domain is not None:
            kwds["box"] = domain
        return MeshClass.from_box(**kwds)
    

class ChipMesh2D(CNodeType):
    
    TITLE: str = "Chip Mesh 2D"
    PATH: str = "mesh.chipmesh"
    INPUT_SLOTS = [
        PortConf("box", DataType.NONE, 0, default = [0.0, 1.0, 0.0, 1.0]),
        PortConf("holes", DataType.NONE, 0, default = [[0.3, 0.3, 0.1], [0.3, 0.7, 0.1], [0.7, 0.3, 0.1], [0.7, 0.7, 0.1]]),
        PortConf("h", DataType.FLOAT, 0, default=0.2, min_val=0.001)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH)
    ]

    @staticmethod
    def run(box, holes, h):
        from fealpy.mesh import  TriangleMesh
        return TriangleMesh.from_box_with_circular_holes(box = box, holes = holes, h = h)

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
    

class Createmesh(CNodeType):
    r"""Create a mesh object.This node generates a mesh of the specified type 
    using given node and cell data.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        Supported values: "triangle", "quadrangle", "tetrahedron", "hexahedron".Default is "edgemesh".
        
        domain (tuple[float, float], optional): Domain.
        node(tensor):Coordinates of mesh nodes.
        cell(tensor):Connectivity of mesh cells.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "CreateMesh"
    PATH: str = "mesh.creation"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, default="edgemesh", 
                 items=["triangle", "quadrangle", "tetrahedron", "hexahedron"]),
        PortConf("domain", DataType.NONE),
        PortConf("node", DataType.TENSOR),
        PortConf("cell", DataType.TENSOR)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH)
    ]

    @staticmethod
    def run(mesh_type, node, cell):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"node": node, "cell": cell}
        return MeshClass(**kwds)
    
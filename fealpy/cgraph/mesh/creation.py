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
    TITLE: str = "二维 Box 网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", items=["triangle", "quadrangle"]),
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("nx", DataType.INT, title="X 分段数", default=10, min_val=1),
        PortConf("ny", DataType.INT, title="Y 分段数", default=10, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, domain, nx, ny):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"nx": nx, "ny": ny}
        if domain is not None:
            kwds["box"] = domain
        return MeshClass.from_box(**kwds)

class SphericalShell3d(CNodeType):
    r"""Create a tetrahedral mesh in a spherical shell region.

    Inputs:
        r1 (float, optional): Inner radius of the spherical shell.
        r2 (float, optional): Outer radius of the spherical shell.
        h (float, optional): Mesh size parameter.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "带空腔的球体网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("r1", DataType.FLOAT, title="内半径", default=0.05, min_val=0.0),
        PortConf("r2", DataType.FLOAT, title="外半径", default=0.5, min_val=0.0),
        PortConf("h", DataType.FLOAT, title="网格尺度", default=0.04, min_val=1e-6),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    @staticmethod
    def run(r1, r2, h):
        from fealpy.mesh import TetrahedronMesh
        mesh = TetrahedronMesh.from_spherical_shell(r1=r1, r2=r2, h=h)
        return mesh

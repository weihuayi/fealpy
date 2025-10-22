
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = ["Cylinder3d"]


class Cylinder3d(CNodeType):
    r"""Create a mesh in a cylinder-shaped 3D area.

    Inputs:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        lc (float, optional): Target mesh size.
    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "圆柱体网格"
    PATH: str = "网格.构造"
    DESC: str = """该节点用于在三维圆柱体区域内生成有限元网格，用户可指定圆柱体的半径、高度及目标网格尺寸，
                以满足不同几何与精度需求的数值模拟。"""
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tetrahedron", items=["tetrahedron"]),
        PortConf("radius", DataType.FLOAT, 1, title="圆柱体半径", default=1.0, min_val=1e-6),
        PortConf("height", DataType.FLOAT, 1, title="圆柱体高度", default=2.0, min_val=1e-6),
        PortConf("lc", DataType.FLOAT, 1, title="网格尺寸", default=0.2, min_val=1e-6)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ]

    @staticmethod
    def run(mesh_type, radius, height, lc): 
        import matplotlib.pyplot as plt
        MeshClass = get_mesh_class(mesh_type)
        mesh = MeshClass.from_cylinder_gmsh(radius=radius, height=height, lc=lc)

        return mesh

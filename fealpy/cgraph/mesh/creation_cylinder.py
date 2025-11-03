
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
    DESC: str = """创建三维空间中，圆柱体区域对应的四面体网格。
                该节点创建圆柱体区域对应的四面体网格，底面圆心半径为原点，半径由输入参数（radius）确定，
                圆柱高度由输入参数（height）设置，使用 gmsh 生成网格，并根据输入的网格尺寸（lc）控制网格密度，
                可以通过输入参数 mesh_type 选择生成的网格类型，但目前只支持四面体网格（tetrahedron）。
                使用例子：通过 MENU 类型的输入参数 mesh_type 选择需要生成的网格类型，
                并向该节点的相应输入槽输入表示底面圆半径（radius）、圆柱高度（height）、网格尺寸（lc）的三个浮点数，
                再将该节点连接到输出，即可查看网格构造效果。
                """
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


from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['Torus2d']


class Torus2d(CNodeType):
    r"""Generate a mesh for a torus.

    Inputs:
        mesh_type (str): Type of mesh to generate.
        R (float): major radius.
        r (float): minor radius.
        x (float): center x coordinate.
        y (float): center y coordinate.
        z (float): center z coordinate.
        h (float): mesh size.
        ax (float): x component of axis direction.
        ay (float): y component of axis direction.
        az (float): z component of axis direction.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "圆环面网格"
    PATH: str = "网格.构造"
    DESC: str = """创建三维空间中，圆环面对应的三角形网格或四边形网格。
                该节点创建圆环面对应的三角形网格或四边形网格，圆环圆心由输入参数（x, y, z）控制，
                圆环内径与外径分别由输入参数 r、R 确定，通过输入参数（ax, ay, az）设置圆环面的轴向（即面法向），
                使用 gmsh 生成网格，并根据输入的网格尺寸（h）控制网格密度，
                可以通过输入参数 mesh_type 选择生成三角形（tri）或者四边形网格（quad）。
                使用例子：通过 MENU 类型的输入参数 mesh_type 选择需要生成的网格类型，
                并向该节点的相应输入槽输入表示圆环圆心坐标的（x, y, z）、外径与内径（R, r）、轴向（ax, ay, az）、网格尺寸（h）九个浮点数，
                再将该节点连接到输出，即可查看网格构造效果。
                """
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tri", items=["tri", "quad"]),
        PortConf("R", DataType.FLOAT, 1, default=1.0, title="长轴"),
        PortConf("r", DataType.FLOAT, 1, default=0.3, title="短轴"),
        PortConf("x", DataType.FLOAT, 1, default=0.0, title="中心X坐标"),
        PortConf("y", DataType.FLOAT, 1, default=0.0, title="中心Y坐标"),
        PortConf("z", DataType.FLOAT, 1, default=0.0, title="中心Z坐标"),
        PortConf("h", DataType.FLOAT, 1, default=0.1, title="网格尺寸"),
        PortConf("ax", DataType.FLOAT, 1, default=0.0, title="轴向X"),
        PortConf("ay", DataType.FLOAT, 1, default=0.0, title="轴向Y"),
        PortConf("az", DataType.FLOAT, 1, default=1.0, title="轴向Z"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ]

    @staticmethod
    def run(mesh_type, R, r, x, y, z, h, ax, ay, az):
        from fealpy.mesher import TorusMesher
        MeshClass = TorusMesher()
        MeshClass.init_mesh.set(mesh_type)
        kwds = {"R": R, "r": r, "x": x, "y": y, "z": z, "h": h, "ax": ax, "ay": ay, "az": az}
        return MeshClass.init_mesh(**kwds)

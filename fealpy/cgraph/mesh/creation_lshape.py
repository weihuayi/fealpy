from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Lshape2d"]


class Lshape2d(CNodeType):
    r"""Create a 2D L-shaped mesh.

    Inputs:
        big_box (tuple[float, float, float, float]): Bounds of the large rectangle (xmin, xmax, ymin, ymax).
        small_box (tuple[float, float, float, float]): Bounds of the rectangle to remove (xmin, xmax, ymin, ymax).
        nx (int): Segments on x direction..
        ny (int): Segments on y direction..
    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "L区域网格"
    PATH: str = "网格.构造"
    DESC: str = """创建二维空间中，L 形区域对应的三角形或四边形网格。
                该节点创建 L 形区域对应的三角形或四边形网格，由输入参数大区域（big_box）和小区域（small_box）确定 L 形区域的范围，
                即大区域与小区域的差集，并使用输入参数（nx, ny）分别设置在 X 方向与 Y 方向的分段数，
                并可以通过输入参数 mesh_type 选择生成的网格类型，三角形（triangle）或四边形网格（quadrangle）。
                使用例子：通过两个“数据.区域”节点分别创建大区域和小区域，连接到该节点的相应输入上，
                并向该节点的相应输入槽输入表示 X 方向与 Y 方向的分段数（nx, ny）的两个整数，
                再将该节点连接到输出，即可查看网格构造效果。
                """
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tri_threshold", items=["triangle", "quadrangle"]),
        PortConf("big_box", DataType.DOMAIN, title="大矩形区域"),
        PortConf("small_box", DataType.DOMAIN, title="挖去小矩形区域"),
        PortConf("nx", DataType.INT, title="X 分段数", default=10, min_val=1),
        PortConf("ny", DataType.INT, title="Y 分段数", default=10, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ]

    @staticmethod
    def run(mesh_type, big_box, small_box, nx, ny): 
        from fealpy.mesher import LshapeMesher
        MeshClass = LshapeMesher()
        MeshClass.init_mesh.set(mesh_type)
        kwds = {"big_box": big_box, "small_box": small_box, 
                "nx": nx, "ny": ny}
        return MeshClass.init_mesh(**kwds)

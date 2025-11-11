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

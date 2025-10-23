
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = ["CreateMesh", "DLDMicrofluidicChipMesh2d"]


class CreateMesh(CNodeType):
    r"""Create a mesh object.This node generates a mesh of the specified type 
    using given node and cell data.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        Supported values: "triangle", "quadrangle", "tetrahedron", "hexahedron".Default is "edgemesh".
        node(tensor):Coordinates of mesh nodes.
        cell(tensor):Connectivity of mesh cells.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "构造网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", 
                 items=["triangle", "quadrangle", "tetrahedron", "hexahedron", "edge"]),
        PortConf("node", DataType.TENSOR, 1, title="节点坐标"),
        PortConf("cell", DataType.TENSOR, 1, title="单元")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, node, cell):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"node": node, "cell": cell}
        return MeshClass(**kwds)


class DLDMicrofluidicChipMesh2d(CNodeType):
    r"""Create a mesh in a DLD microfluidic chip-shaped 2D area.

    Inputs:
        init_point X (float, optional): Initial point of the chip.
        init_point Y (float, optional): Initial point of the chip.
        chip_height (float, optional): Height of the chip.
        inlet_length (float, optional): Length of the inlet.
        outlet_length (float, optional): Length of the outlet.
        radius (float, optional): Radius of the micropillars.
        n_rows (int, optional): Number of rows of micropillars.
        n_cols (int, optional): Number of columns of micropillars.
        tan_angle (float, optional): Tangent value of the angle of deflection.
        n_stages (int, optional): Number of periods of micropillar arrays.
        stage_length (float, optional): Length of a single period.
        lc (float, optional): Target mesh size.

    Outputs:
        mesh (MeshType): The mesh object created.
        radius (float): Radius of the micropillars.
        centers (ndarray): Coordinates of the centers of the micropillars.
        inlet_boundary (ndarray): Inlet boundary.
        outlet_boundary (ndarray): Outlet boundary.
        wall_boundary (ndarray): Wall boundary of the channel.
    """
    TITLE: str = "二维 DLD 微流芯片网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("init_point_x", DataType.FLOAT, 1, default=0.0, title="初始点 X"),
        PortConf("init_point_y", DataType.FLOAT, 1, default=0.0, title="初始点 Y"),
        PortConf("chip_height", DataType.FLOAT, 1, default=1.0, title="芯片长度"),
        PortConf("inlet_length", DataType.FLOAT, 1, default=0.1, title="入口宽度"),
        PortConf("outlet_length", DataType.FLOAT, 1, default=0.1, title="出口宽度"),
        PortConf("radius", DataType.FLOAT, 1, default=1 / (3 * 4 * 3), title="微柱半径"),
        PortConf("n_rows", DataType.INT, 1, default=8, title="行数"),
        PortConf("n_cols", DataType.INT, 1, default=4, title="列数"),
        PortConf("tan_angle", DataType.FLOAT, 1, default=1/7, title="偏转角正切值"),
        PortConf("n_stages", DataType.INT, 1, default=3, title="微柱阵列周期数"),
        PortConf("stage_length", DataType.FLOAT, 1, default=1.4, title="单周期长度"),
        PortConf("lc", DataType.FLOAT, 1, default=0.02, title="网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱圆心坐标"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.geometry import DLDMicrofluidicChipModeler
        from fealpy.mesher import DLDMicrofluidicChipMesher
        import gmsh

        options = {
            "init_point" : (options.get("init_point_x"), options.get("init_point_y")),
            "chip_height" : options.get("chip_height"),
            "inlet_length" : options.get("inlet_length"),
            "outlet_length" : options.get("outlet_length"),
            "radius" : options.get("radius"),
            "n_rows" : options.get("n_rows"),
            "n_cols" : options.get("n_cols"),
            "tan_angle" : options.get("tan_angle"),
            "n_stages" : options.get("n_stages"),
            "stage_length" : options.get("stage_length"),
            "lc" : options.get("lc")
        }

        gmsh.initialize()
        modeler = DLDMicrofluidicChipModeler(options)
        modeler.build(gmsh)
        mesher = DLDMicrofluidicChipMesher(options)
        mesher.generate(modeler, gmsh)
        gmsh.finalize()

        return (mesher.mesh, mesher.radius, mesher.centers, mesher.inlet_boundary, 
                mesher.outlet_boundary, mesher.wall_boundary)

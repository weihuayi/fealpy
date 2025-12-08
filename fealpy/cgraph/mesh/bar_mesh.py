from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25Mesh", 
           "Bar942Mesh",
           "TrussTowerMesh"]

class Bar25Mesh(CNodeType):
    r"""Generate a 3D mesh for a 25-bar truss structure.

    This node generates the node coordinates and cell connectivity for a 
    classic 25-bar space truss.
        
    Outputs:
        mesh (Mesh): The generated 25-bar truss mesh.
    """
    TITLE: str = "25杆桁架网格"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="25杆桁架网格", title="网格")
    ]

    @staticmethod
    def run():
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        
        node = bm.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540],
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0],
            [-2540, -2540, 0]], dtype=bm.float64)
        cell = bm.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4],
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5],
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=bm.int32)
        
        mesh = EdgeMesh(node, cell)

        return mesh
    
class Bar942Mesh(CNodeType):
    r"""Generate a 3D mesh for a 942-bar truss structure.

    This node generates the node coordinates and cell connectivity for a 
    classic 942-bar space truss.

    Inputs:
        d1 (float): Half-width of the first layer.
        d2 (float): Width of the second layer.
        d3 (float): Width of the third layer.
        d4 (float): Width of the fourth layer.
        r2 (float): Radius of the second layer.
        r3 (float): Radius of the third layer.
        r4 (float): Radius of the fourth layer.
        l3 (float): Height of the third segment.
        l2 (float): Height of the second segment.
        l1 (float): Height of the first segment.

    Outputs:
        mesh (Mesh): The generated 942-bar truss mesh.
    """
    TITLE: str = "942杆桁架网格"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS = [
        PortConf("d1", DataType.FLOAT, 1, desc="第一层半宽(正方形顶部)", title="第一层半宽", default=2135.0),
        PortConf("d2", DataType.FLOAT, 1, desc="第二层宽度(八边形段)", title="第二层宽度", default=5335.0),
        PortConf("d3", DataType.FLOAT, 1, desc="第三层宽度(十二边形段)", title="第三层宽度", default=7470.0),
        PortConf("d4", DataType.FLOAT, 1, desc="第四层宽度(底层支座)", title="第四层宽度", default=9605.0),
        PortConf("r2", DataType.FLOAT, 1, desc="第二层半径(八边形段)", title="第二层半径", default=4265.0),
        PortConf("r3", DataType.FLOAT, 1, desc="第三层半径(十二边形段)", title="第三层半径", default=6400.0),
        PortConf("r4", DataType.FLOAT, 1, desc="第四层半径(底层支座)", title="第四层半径", default=8535.0),
        PortConf("l3", DataType.FLOAT, 1, desc="第三段高度(十二边形段总高)", title="第三段高度", default=43890.0),
        PortConf("l2", DataType.FLOAT, 0, desc="第二段高度(八边形段顶部高度，默认l3+29260)", title="第二段高度", default=None),
        PortConf("l1", DataType.FLOAT, 0, desc="第一段高度(正方形段顶部高度，默认l2+21950)", title="第一段高度", default=None)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="924杆桁架网格", title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.csm.mesh.bar942 import Bar942
        from fealpy.mesh import EdgeMesh
        bar = Bar942()
        nodes, cells = bar.build_truss_3d(
            d1=options.get("d1"),
            d2=options.get("d2"),
            d3=options.get("d3"),
            d4=options.get("d4"),
            r2=options.get("r2"),
            r3=options.get("r3"),
            r4=options.get("r4"),
            l3=options.get("l3"),
            l2=options.get("l2"),
            l1=options.get("l1")
        )
        mesh = EdgeMesh(nodes, cells)
        return mesh

class TrussTowerMesh(CNodeType):
    r"""Generate a 3D mesh for a truss tower structure.

    Inputs:
        n_panel (int): Number of vertical panels.
        Lz (float): Total height in the z-direction.
        Wx (float): Half-length in the x-direction.
        Wy (float): Half-length in the y-direction.
        lc (float): Characteristic length for mesh size control.
        ne_per_bar (int): Number of elements per bar.
        face_diag (bool): Whether to include face diagonal bracing.
        
    Outputs:
        mesh (Mesh): The generated 3D truss tower mesh.
    """
    TITLE: str = "桁架塔网格"
    PATH: str = "preprocess.mesher"       
    INPUT_SLOTS = [
        PortConf("n_panel", DataType.INT, 1, desc="沿 z 方向的面板数量（≥1）", title="面板数量", default=19),
        PortConf("Lz", DataType.FLOAT, 1, desc="桁架塔沿 z 方向的总长度", title="总长度", default=19.0),
        PortConf("Wx", DataType.FLOAT, 1, desc="截面矩形的 x 方向半宽度", title="截面宽度", default=0.45),
        PortConf("Wy", DataType.FLOAT, 1, desc="截面矩形的 y 方向半宽度", title="截面高度", default=0.40),
        PortConf("lc", DataType.FLOAT, 1, desc="用于控制网格尺寸的几何特征长度", title="几何点特征长度", default=0.1),
        PortConf("ne_per_bar", DataType.INT, 1, desc="每根杆件沿长度方向划分的单元数量（≥1）", title="每根杆件单元数", default=1),
        PortConf("face_diag", DataType.BOOL, 0, desc="是否在四个侧面加入面内对角线加劲杆件（默认True）", title="面内对角加劲", default=True)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="生成桁架塔网格", title="网格"),
        PortConf("NN", DataType.INT, desc="桁架塔节点数量", title="节点数量"),
        PortConf("NC", DataType.INT, desc="桁架塔单元总数", title="单元数量")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.mesh.truss_tower import TrussTower

        node, cell = TrussTower.build_truss_3d_zbar(
            n_panel=options.get("n_panel"),
            Lz=options.get("Lz"),
            Wx=options.get("Wx"),
            Wy=options.get("Wy"),
            lc=options.get("lc"),
            ne_per_bar=options.get("ne_per_bar"),
            face_diag=options.get("face_diag"),
            save_msh=None
        )
        
        from fealpy.mesh import EdgeMesh
        mesh = EdgeMesh(node, cell)
        return mesh

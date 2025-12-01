from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ChannelBeamMesh", 
           "TimobeamAxleMesh"]

class ChannelBeamMesh(CNodeType):
    r"""3D Channel Beam Mesh Generator.
    
    The node generates a one-dimensional mesh along the length of a 3D channel beam.
    
    Inputs:
        L (FLOAT): Length of the beam (m). Default 1.0.
        n_ele (INT): Number of elements along the beam length. Default 10.
        
    Outputs:
        mesh (MESH): Generated 1D mesh along the beam length.
        NN (INT): Number of nodes in the mesh.
        NC (INT): Number of cells in the mesh.
    """
    
    TITLE: str = "槽形梁网格生成"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点用于生成三维槽形梁的一维网格。"""
    
    INPUT_SLOTS = [
        PortConf("L", DataType.FLOAT, 0, desc="梁的长度", title="梁长度", default=1.0),
        PortConf("n_ele", DataType.INT, 0, desc="沿梁长度方向的单元数量", 
                 title="单元数量", default=10),
    ]
        
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="槽形梁网格", title="网格"),
        PortConf("NN", DataType.INT, desc="槽形梁的节点数量", title="节点数量"),
        PortConf("NC", DataType.INT, desc="槽形梁的单元数量", title="单元数量")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesh.edge_mesh import EdgeMesh
        L = options.get("L")
        n = options.get("n_ele")
        
        nodes = bm.linspace(0, L, n + 1)
        node = bm.zeros((n + 1, 3), dtype=bm.float64)
        node[:, 0] = nodes  # x-coordinates along beam length
        
        cell = bm.zeros((n, 2), dtype=bm.int32)
        cell[:, 0] = bm.arange(n)
        cell[:, 1] = bm.arange(1, n + 1)
        
        mesh = EdgeMesh(node, cell)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        
        return mesh, NN, NC
    

class TimobeamAxleMesh(CNodeType):
    r"""3D Timoaxle Beam Mesh Generator.
    
    The node generates a one-dimensional mesh for a 3D Timoaxle beam.
    
    Inputs:
        None.
        
    Outputs:
        mesh (MESH): Generated 1D mesh for the Timoaxle beam.
        NN (INT): Number of nodes in the mesh.
        NC (INT): Number of cells in the mesh.
    """
    
    TITLE: str = "列车轮轴网格生成"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点用于生成列车轮轴的一维网格。"""

    INPUT_SLOTS = []

    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="列车轮轴网格", title="网格"),
        PortConf("NN", DataType.INT, desc="列车轮轴的节点数量", title="节点数量"),
        PortConf("NC", DataType.INT, desc="列车轮轴的单元数量", title="单元数量")
    ]

    @staticmethod
    def run():
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        
        node = bm.array([[0.0, 0.0, 0.0], [70.5, 0.0, 0.0], [141.0, 0.0, 0.0], [155.0, 0.0, 0.0],
            [169.0, 0.0, 0.0], [213.25, 0.0, 0.0], [257.5, 0.0, 0.0], [301.75, 0.0, 0.0],
            [346.0, 0.0, 0.0], [480.0, 0.0, 0.0], [614.0, 0.0, 0.0], [853.0, 0.0, 0.0],
            [1092.0, 0.0, 0.0], [1334.0, 0.0, 0.0], [1576.0, 0.0, 0.0], [1620.25, 0.0, 0.0],
            [1664.5, 0.0, 0.0], [1708.75, 0.0, 0.0], [1753.0, 0.0, 0.0], [1767.0, 0.0, 0.0],
            [1781.0, 0.0, 0.0], [1851.5, 0.0, 0.0],[1922.0, 0.0, 0.0], [169.0, 0.0, -100.0],
            [213.25, 0.0, -100.0], [257.5, 0.0, -100.0], [301.75, 0.0, -100.0], [346.0, 0.0, -100.0],
            [1576.0, 0.0, -100.0], [1620.25, 0.0, -100.0], [1664.5, 0.0, -100.0], 
            [1708.75, 0.0, -100.0], [1753.0, 0.0, -100.0]], dtype=bm.float64)

        cell = bm.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
             [5, 6], [6, 7], [7, 8],[8, 9], [9, 10], [10, 11],
             [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
             [16, 17],[17, 18], [18, 19], [19, 20], [20, 21], 
             [21, 22], [4, 23], [5, 24], [6, 25], [7, 26], [8, 27], 
             [14, 28], [15, 29],[16, 30], [17, 31], [18,32]], dtype=bm.int32)
        
        mesh = EdgeMesh(node, cell)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        return mesh, NN, NC
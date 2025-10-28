from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Truss3dData"]

class Truss3dData(CNodeType):
    r"""A 25-bar truss structure model.

    This node generates the node coordinates and cell connectivity for a 
    classic 25-bar space truss.

    Outputs:
        node (tensor): The node coordinates of the truss.
        cell (tensor): The cell connectivity of the truss (edges).
    """
    TITLE: str = "25杆桁架几何"
    PATH: str = "模型.几何"
    DESC: str = "生成一个经典的25杆空间桁架的节点坐标和单元连接关系。"

    INPUT_SLOTS = []

    OUTPUT_SLOTS = [
        PortConf("node", DataType.TENSOR, desc="桁架的节点坐标", title="节点坐标"),
        PortConf("cell", DataType.TENSOR, desc="桁架的单元连接关系", title="单元"),
    ]

    @staticmethod
    def run():
        from fealpy.backend import backend_manager as bm

        node = bm.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540],
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0],
            [-2540, -2540, 0]], dtype=bm.float64)
        edge = bm.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4],
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5],
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=bm.int32)
        
        cell = edge

        return node, cell
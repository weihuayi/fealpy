from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Truss3dData"]

class Truss3dData(CNodeType):
    r"""
    Data node for the 25-bar 3D truss benchmark problem.

    This node generates the edge mesh, global load vector, and Dirichlet boundary indices
    for the truss structure based on user-specified parameters.

    Inputs (parameter slots):
        p (float): Concentrated force along the y-axis (default: 900.0).
        top_z (float): Z-coordinate of the loaded nodes (default: 5080.0).
        fixed_nodes (list[int]): Node indices with fixed displacement (default: [6, 7, 8, 9]).

    Outputs:
        mesh (Mesh): Edge mesh object for the truss.
        F (tensor): Global load vector (shape: 3*NN,).
        fixed_dofs (tensor): Dirichlet boundary degree-of-freedom indices.

    Note:
        - This is a data source node; it does not require upstream inputs.
        - All outputs are concrete values, not function objects.
    """
    TITLE: str = "桁架问题数据"
    PATH: str = "网格.构造"
    DESC: str = "根据参数生成25杆桁架的网格、外载荷向量与边界自由度索引。"

    INPUT_SLOTS = [
        PortConf("p", DataType.FLOAT, 0, desc="沿y轴的集中力", title="载荷"),
        PortConf("top_z", DataType.FLOAT, 0, desc="受力层Z坐标", title="受力层Z"),
        PortConf("fixed_nodes", DataType.TENSOR, 0, desc="固定约束的节点", title="固定节点"),
    ]

    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="桁架的边网格对象", title="桁架网格"),
        PortConf("F", DataType.TENSOR, desc="全局外载荷向量", title="外载荷向量"),
        PortConf("fixed_dofs", DataType.TENSOR, desc="Dirichlet自由度索引", title="边界自由度索引"),
    ]

    @staticmethod
    def run(p=None, top_z=None, fixed_nodes=None):
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import EdgeMesh
        import re
        def to_float(x, d): 
            if x is None or (isinstance(x, str) and x.strip() == ""): return float(d)
            return float(x)
        def to_int_list(x, d):
            if x is None: return list(d)
            if isinstance(x, (list, tuple)): return [int(v) for v in x]
            if isinstance(x, str):
                parts = [s for s in re.split(r"[,\s]+", x.strip()) if s]
                return [int(s) for s in parts] if parts else list(d)
            return [int(x)]

        p = to_float(p, 900.0)
        top_z = to_float(top_z, 5080.0)
        fixed_nodes = to_int_list(fixed_nodes, (6, 7, 8, 9))

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
        mesh = EdgeMesh(node, edge)

        node_coords = mesh.entity("node")
        GD, NN = 3, mesh.number_of_nodes()
        F = bm.zeros((NN, GD), dtype=bm.float64)
        idx = bm.where(node_coords[..., 2] == top_z)[0]
        if idx.size > 0:
            F[idx] = bm.array([0.0, p, 0.0], dtype=bm.float64)
        F = F.reshape(-1)

        fn = bm.asarray(fixed_nodes, dtype=bm.int32)
        fixed_dofs = bm.concatenate([3*fn + k for k in range(3)]).astype(bm.int32)

        return mesh, F, fixed_dofs

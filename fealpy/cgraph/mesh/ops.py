from ..nodetype import CNodeType, PortConf, DataType


class MeshDimensionUpgrading(CNodeType):
    """Geometric 2d to 3d"""
    TITLE = "网格升维"
    PATH = "mesh.ops"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="2D 网格"),
        PortConf("z", DataType.TENSOR, 1, desc="网格顶点在 Z 方向的坐标", title="Z 位置")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="3D 网格")
    ]

    @staticmethod
    def run(mesh, z):
        from fealpy.backend import bm
        assert mesh.geo_dimension() == 2
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        new_node = bm.concat([node, z[:, None]], axis=1)
        # TODO: avoid unnecessary construction
        return mesh.__class__(new_node, cell)


class ErrorEstimation(CNodeType):
    r"""Error estimator on mesh.

    Args:
        celltype (bool, optional): Return error on each cell if `True`. Defaults to `False`.

    Inputs:
        mesh (SupportsError): The mesh to estimate error on.
        u (Function | Callable):
        v (Function | Callable):
        q (int, optional): Index of quadrature formula.
        power (int, optional):

    Outputs:
        out (Tensor): Error between u and v.
    """
    TITLE = "误差估计"
    PATH = "mesh"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("u", DataType.FUNCTION),
        PortConf("v", DataType.FUNCTION),
        PortConf("q", DataType.INT, title="积分公式", default=0, min_val=0, max_val=17),
        PortConf("power", DataType.INT, title="次数", default=2, min_val=0),
        PortConf("celltype", DataType.BOOL, title="每个单元", default=False)
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="误差")
    ]

    @staticmethod
    def run(mesh, u, v, q, power, celltype):
        return mesh.error(u, v, q=q, power=power, celltype=celltype)

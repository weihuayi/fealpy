from ..nodetype import CNodeType, PortConf, DataType


class Error(CNodeType):
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
    TITLE = "Error Estimation"
    PATH = "mesh"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH),
        PortConf("u", DataType.FUNCTION),
        PortConf("v", DataType.FUNCTION),
        PortConf("q", DataType.INT, default=0, min_val=0, max_val=17),
        PortConf("power", DataType.INT, default=2, min_val=0),
        PortConf("celltype", DataType.BOOL, default=False)
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR)
    ]

    @staticmethod
    def run(mesh, u, v, q, power, celltype):
        return mesh.error(u, v, q=q, power=power, celltype=celltype)

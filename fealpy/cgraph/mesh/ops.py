from ..core import Node


class Error(Node):
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
    def __init__(
            self,
            *,
            celltype: bool = False
    ):
        super().__init__()
        self.cell_type = celltype
        self.kwargs = {'celltype': celltype}
        self.register_input("mesh")
        self.register_input("u")
        self.register_input("v")
        self.register_input("q", default=3)
        self.register_input("power", default=2)
        self.register_output("out")

    def run(self, mesh, u, v, q, power):
        return mesh.error(u, v, q=q, power=power, **self.kwargs)

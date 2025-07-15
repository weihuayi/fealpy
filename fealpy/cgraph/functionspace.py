from typing import Type
import importlib

from ..functionspace import Function
from .core import Node


SPACE_CLASSES = {
    "bernstein": ("bernstein_fe_space", "BernsteinFESpace"),
    "lagrange": ("lagrange_fe_space", "LagrangeFESpace")
}


def get_space_class(space_type: str) -> Type:
    m = importlib.import_module(
        f"fealpy.functionspace.{SPACE_CLASSES[space_type][0]}"
    )
    return getattr(m, SPACE_CLASSES[space_type][1])


class FunctionSpace(Node):
    def __init__(self, space_type: str, ctype: str = "C"):
        super().__init__()
        self.SpaceClass = get_space_class(space_type)
        self.register_input("mesh")
        self.register_input("p", default=1)
        self.register_output("space")
        self.kwargs = {"ctype": ctype}

    def run(self, mesh, p):
        return self.SpaceClass(mesh, p, **self.kwargs)


class TensorToFEFunction(Node):
    def __init__(self, coordtype: str = 'barycentric'):
        super().__init__()
        self.coordtype = coordtype
        self.register_input("tensor")
        self.register_input("space")
        self.register_output("out")

    def run(self, tensor, space):
        return Function(space, tensor, coordtype=self.coordtype)

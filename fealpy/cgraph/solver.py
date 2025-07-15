
from ..backend import TensorLike as _DT
from ..solver import cg
from .core import Node


class CGSolver(Node):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self.register_input("A")
        self.register_input("b")
        self.register_input("x0", default=None)
        self.register_output("out")

    def run(self, A, b, x0=None) -> _DT:
        return cg(A, b, x0, **self._kwargs)

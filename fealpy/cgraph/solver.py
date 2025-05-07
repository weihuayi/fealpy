
from ..backend import TensorLike as _DT
from ..solver import cg
from .base import Node


class CGSolver(Node):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self.add_input("A")
        self.add_input("b")
        self.add_input("x0", default=None)
        self.add_output("out")

    def run(self, A, b, x0=None) -> _DT:
        return cg(A, b, x0, **self._kwargs)

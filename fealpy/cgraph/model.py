
from ..functionspace import Function
from .base import Node


class Model(Node):
    def __init__(self):
        super().__init__()

    ### Utils

    def linear_system(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, A, F, solver: str):
        if solver == 'cg':
            from ..solver import cg
            return cg(A, F)
        # TODO: add more solvers ...

    ### Methods

    def run_single(self, space, *args, **kwargs):
        A, F = self.linear_system(*args, **kwargs)
        uh = self.solve(A, F, self.solver)
        return Function(space, uh, "barycentric")

    def run_error(self, space, pde, *args, **kwargs):
        uh = self.run_single(space, *args, **kwargs)
        mesh = space.mesh
        err = mesh.error(pde.solution, uh)
        return err

    # TODO: add more methods ...
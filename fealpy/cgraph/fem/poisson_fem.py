
from ...fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from ..base import Node


class PoissonFEM(Node):
    def __init__(self, di_method='fast', si_method=None):
        super().__init__()
        self.di_method = di_method
        self.si_method = si_method
        self.run = self.linear_system
        self.add_input("space")
        self.add_input("gd")
        self.add_input("source", default=None)
        self.add_input("q", default=None)
        self.add_output("A")
        self.add_output("F")

    def linear_system(self, space, gd, source, q: int):
        DI = ScalarDiffusionIntegrator(q=q, method=self.di_method)
        bform = BilinearForm(space) << DI
        SI = ScalarSourceIntegrator(source, q=q, method=self.si_method)
        lform = LinearForm(space) << SI
        dbc = DirichletBC(space, gd=gd)
        return dbc.apply(bform.assembly(), lform.assembly())

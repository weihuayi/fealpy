from typing import Tuple, TypeVar 

Field = TypeVar('Field')

class Quadrature():
    r"""Base class for quadrature generators."""
    def number_of_quadrature_points(self) -> int:
        return self.NQ

    def get_quadrature_points_and_weights(self) -> Tuple[Field, Field]:
        return self.quadpts, self.weights 

    def get_quadrature_precision(self) -> int:
        return self.order


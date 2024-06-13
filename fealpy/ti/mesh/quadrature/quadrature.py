from typing import Dict, Optional, Union, Tuple

import taichi as ti 

from ti.types import template as Template 


class Quadrature():
    r"""Base class for quadrature generators."""
    def number_of_quadrature_points(self) -> int:
        return self.NQ

    def get_quadrature_points_and_weights(self) -> Template:
        return self.quadpts, self.weights 

    def get_quadrature_points_and_weights(self) -> Tuple[Template, Template]:
        return self.quadpts, self.weights 

    def get_quadrature_precision(self) -> int:
        return self.order


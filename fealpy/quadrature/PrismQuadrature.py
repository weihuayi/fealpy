import numpy as np
from .Quadrature import Quadrature
from .IntervalQuadrature import IntervalQuadrature
from .TriangleQuadrature import TriangleQuadrature


class PrismQuadrature(Quadrature):
    def __init__(self, index):
        q0 = IntervalQuadrature(index)
        q1 = TriangleQuadrature(index)
        n0 = q0.number_of_quadrature_points()
        n1 = q1.number_of_quadrature_points()
        bc0, ws0 = 

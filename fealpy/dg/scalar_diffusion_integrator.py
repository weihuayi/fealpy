
import numpy as np

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalarDiffusionIntegrator():
    def __init__(self, q: int) -> None:
        self.q = q

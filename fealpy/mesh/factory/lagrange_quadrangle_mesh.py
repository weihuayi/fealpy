from .base import MeshFactory


class LagrangeQuadrangleMesh(metaclass=MeshFactory):
    schema = "lagrange_quad"
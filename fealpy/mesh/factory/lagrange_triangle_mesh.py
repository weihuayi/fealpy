from .base import MeshFactory


class LagrangeTriangleMesh(metaclass=MeshFactory):
    schema = "lagrange_tri"
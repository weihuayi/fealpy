from .base import MeshFactory


class PolygonMesh(metaclass=MeshFactory):
    schema = "poly"
from typing import Tuple, Type
import importlib
from ..base import Node


def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)


class Box2d(Node):
    r"""Create a mesh in a box-shaped 2D area.

    Args:
        mesh_type (str): Type of mesh to granerate.
        itype (dtype | None, optional): Scalar type of integers.
        ftype (dtype | None, optional): Scalar type of floats.
        device (device | None, optional): Device.

    Inputs:
        xmin, xmax, ymin, ymax (float, optional): Area.
        nx (int, optional): Segments on x direction.
        ny (int, optional): Segments on y direction.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    def __init__(
        self,
        mesh_type: str,
        *,
        itype=None,
        ftype=None,
        device=None
    ):
        super().__init__()
        self.MeshClass = get_mesh_class(mesh_type)
        self.kwargs = {"device": device, "itype": itype, "ftype": ftype}
        self.add_input("xmin", default=0.)
        self.add_input("xmax", default=1.)
        self.add_input("ymin", default=0.)
        self.add_input("ymax", default=1.)
        self.add_input("nx", default=10)
        self.add_input("ny", default=10)
        self.add_output("mesh")

    def run(self, xmin, xmax, ymin, ymax, nx, ny):
        domain = [xmin, xmax, ymin, ymax]
        return self.MeshClass.from_box(domain, nx=nx, ny=ny, **self.kwargs)
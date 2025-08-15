from typing import Type

from .nodetype import CNodeType, PortConf, DataType

__all__ = [
    "FunctionSpace",
    "BoundaryDof",
    "FEFunction"
]

SPACE_CLASSES = {
    "bernstein": ("bernstein_fe_space", "BernsteinFESpace"),
    "lagrange": ("lagrange_fe_space", "LagrangeFESpace")
}


def get_space_class(space_type: str) -> Type:
    import importlib
    m = importlib.import_module(
        f"fealpy.functionspace.{SPACE_CLASSES[space_type][0]}"
    )
    return getattr(m, SPACE_CLASSES[space_type][1])


class FunctionSpace(CNodeType):
    TITLE: str = "Function Space"
    PATH: str = "space.creation"
    INPUT_SLOTS = [
        PortConf("type", DataType.MENU, 0, param="space_type", default="lagrange", items=["lagrange", "bernstein"]),
        PortConf("mesh", DataType.MESH, 1),
        PortConf("p", DataType.INT, 1)
    ]
    OUTPUT_SLOTS = [
        PortConf("space", DataType.SPACE)
    ]

    @staticmethod
    def run(space_type: str, mesh, p):
        SpaceClass = get_space_class(space_type)
        return SpaceClass(mesh, p)


class BoundaryDof(CNodeType):
    TITLE: str = "Boundary Dof"
    PATH: str = "space.ops"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE),
    ]
    OUTPUT_SLOTS = [
        PortConf("isDDof", DataType.TENSOR)
    ]

    @staticmethod
    def run(space):
        return space.is_boundary_dof()


class FEFunction(CNodeType):
    TITLE: str = "FE Function"
    PATH: str = "space"
    INPUT_SLOTS = [
        PortConf("tensor", DataType.TENSOR),
        PortConf("space", DataType.SPACE),
        PortConf("coordtype", DataType.MENU, 0, items=['barycentric', 'cartesian']),
    ]
    OUTPUT_SLOTS = [
        PortConf("function", DataType.FUNCTION)
    ]

    @staticmethod
    def run(tensor, space, coordtype):
        from ..functionspace import Function
        return Function(space, tensor, coordtype=coordtype)

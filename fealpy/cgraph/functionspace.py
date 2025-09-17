from typing import Type

from .nodetype import CNodeType, PortConf, DataType

__all__ = [
    "TensorFunctionSpace",
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
    TITLE: str = "标量函数空间"
    PATH: str = "函数空间.构造"
    INPUT_SLOTS = [
        PortConf("type", DataType.MENU, 0, title="空间类型", param="space_type", default="lagrange", items=["lagrange", "bernstein"]),
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        PortConf("p", DataType.INT, 1, title="次数", default=1, min_val=1, max_val=10)
    ]
    OUTPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间")
    ]

    @staticmethod
    def run(space_type: str, mesh, p):
        SpaceClass = get_space_class(space_type)
        return SpaceClass(mesh, p)


class TensorFunctionSpace(CNodeType):
    TITLE: str = "张量函数空间"
    PATH: str = "函数空间.构造"
    INPUT_SLOTS = [
        PortConf("type", DataType.MENU, 0, title="空间类型", param="space_type", default="lagrange", items=["lagrange", "bernstein"]),
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        PortConf("p", DataType.INT, 1, title="次数", default=1, min_val=1, max_val=10),
        PortConf("gd", DataType.INT, 1, title="自由度长度", param="GD", default=2)
        # PortConf("value_dim", DataType.INT, 1, title="", param="VD", default=-1)
    ]
    OUTPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间")
    ]

    @staticmethod
    def run(space_type: str, mesh, p: int, GD: int):
        from ..functionspace import functionspace
        element = (space_type.capitalize(), p)
        shape = (GD, -1)
        return functionspace(mesh, element, shape=shape)


class BoundaryDof(CNodeType):
    TITLE: str = "边界自由度"
    PATH: str = "函数空间.操作"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间"),
    ]
    OUTPUT_SLOTS = [
        PortConf("isDDof", DataType.TENSOR, title="边界自由度")
    ]

    @staticmethod
    def run(space):
        return space.is_boundary_dof()


class FEFunction(CNodeType):
    TITLE: str = "有限元函数"
    PATH: str = "函数空间.函数"
    INPUT_SLOTS = [
        PortConf("tensor", DataType.TENSOR, title="自由度"),
        PortConf("space", DataType.SPACE, title="函数空间"),
        PortConf("coordtype", DataType.MENU, 0, title="参数类型", items=['barycentric', 'cartesian']),
    ]
    OUTPUT_SLOTS = [
        PortConf("function", DataType.FUNCTION, title="函数")
    ]

    @staticmethod
    def run(tensor, space, coordtype):
        from ..functionspace import Function
        return Function(space, tensor, coordtype=coordtype)

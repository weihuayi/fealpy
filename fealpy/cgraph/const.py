
from .nodetype import CNodeType, PortConf, DataType

__all__ = [
    "ConstDomain",
    "ConstInt",
    "ConstFloat",
    "ConstTensor",
]


def _identity(*args, **kwargs):
    return args + tuple(kwargs.values())


class ConstDomain(CNodeType):
    TITLE = "区域"
    PATH = "数据.常量"
    DESC = "输出一个区域"
    INPUT_SLOTS = [
        PortConf("value", DataType.STRING, 0, title="值", default="[0, 1]")
    ]
    OUTPUT_SLOTS = [
        PortConf("value", DataType.DOMAIN, title="区域")
    ]
    @staticmethod
    def run(value):
        domain = eval(value)
        if isinstance(domain, (list, tuple)) and all(isinstance(x, int) for x in domain):
            return domain
        else:
            raise ValueError("Invalid domain value.")


class ConstInt(CNodeType):
    TITLE = "整数"
    PATH = "数据.常量"
    DESC = "输出一个整数"
    INPUT_SLOTS = [
        PortConf("value", DataType.INT, 0, title="值", default=0)
    ]
    OUTPUT_SLOTS = [
        PortConf("value", DataType.INT, title="整数")
    ]
    run = staticmethod(_identity)


class ConstFloat(CNodeType):
    TITLE = "浮点数"
    PATH = "数据.常量"
    DESC = "输出一个浮点数"
    INPUT_SLOTS = [
        PortConf("value", DataType.FLOAT, 0, title="值", default=0)
    ]
    OUTPUT_SLOTS = [
        PortConf("value", DataType.FLOAT, title="浮点数")
    ]
    run = staticmethod(_identity)


class ConstTensor(CNodeType):
    TITLE = "张量"
    PATH = "数据.常量"
    DESC = "输出一个张量"
    INPUT_SLOTS = [
        PortConf("dtype_name", DataType.MENU, 0, title="数据类型", default="float64", items=['float64', 'float32', 'int64', 'int32', 'bool']),
        PortConf("value", DataType.TEXT, 0, title="值")
    ]
    OUTPUT_SLOTS = [
        PortConf("value", DataType.TENSOR, title="张量")
    ]
    @staticmethod
    def run(dtype_name: str, value):
        import math
        from fealpy.backend import bm
        if dtype_name.startswith(("float", "int", "bool", "uint", "complex"), 0):
            dtype = getattr(bm, dtype_name)
        else:
            raise ValueError(f'{dtype_name} is not supported.')
        return bm.tensor(eval(value, None, vars(math)), dtype=dtype)

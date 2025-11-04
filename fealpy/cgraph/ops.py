from .nodetype import CNodeType, PortConf, DataType


class TensorCompute(CNodeType):
    """Excute a tensor operation.

    Inputs:
        op (str): Tensor operation to execute.
        a (Tensor): First tensor.
        b (Tensor): Second tensor.

    Outputs:
        tensor (Tensor): Output tensor.
    """
    TITLE: str = "张量运算"
    PATH: str = "工具"
    DESC: str = "执行指定的张量运算。"
    INPUT_SLOTS = [
        PortConf(
            "op",
            dtype=DataType.MENU,
            ttype=None,
            desc="执行的张量操作",
            title="张量操作",
            param="op",
            default="相加",
            items=["相加", "相减", "相乘", "相除", "指数"]
        ),
        PortConf(
            "a",
            dtype=DataType.TENSOR,
            ttype=1,
            desc="输入张量 1",
            title="张量",
            param="a",
            default=None
        ),
        PortConf(
            "b",
            dtype=DataType.TENSOR,
            ttype=1,
            desc="输入张量 2",
            title="张量",
            param="b",
            default=None
        )
    ]
    OUTPUT_SLOTS = [
        PortConf("out", dtype=DataType.TENSOR, desc="张量运算结果", title="张量")
    ]

    @staticmethod
    def run(op, a, b):
        from fealpy.backend import bm
        if op == "相加":
            return a + b
        elif op == "相减":
            return a - b
        elif op == "相乘":
            return a * b
        elif op == "相除":
            return a / b
        elif op == "指数":
            return bm.pow(a, b)
        elif op == "叉乘":
            assert a.ndim == 2 and b.ndim == 2
            assert a.shape[1] == 3 and b.shape[1] == 3
            return bm.cross(a, b)
        else:
            raise NotImplementedError(f"op {op} is not implemented.")


class TensorMix(CNodeType):
    """Execute tensor mixing operation.

    Inputs:
        op (str): Tensor operation to execute.
        coef (float): Mix coefficient.
        a (Tensor): First tensor.
        b (Tensor): Second tensor.

    Outputs:
        tensor (Tensor): Output tensor.
    """
    TITLE: str = "张量混合"
    PATH: str = "工具"
    DESC: str = "按比例线性混合两个张量： a * (1 - coef) + b * coef"
    INPUT_SLOTS = [
        PortConf("coef", DataType.FLOAT, 1, desc="混合系数", title="混合系数", default=0.5),
        PortConf("clip", DataType.BOOL, 1, desc="是否将系数控制在 [0, 1] 内", title="钳制系数", default=False),
        PortConf("a", DataType.TENSOR, 1, desc="第一个张量", title="张量"),
        PortConf("b", DataType.TENSOR, 1, desc="第二个张量", title="张量"),
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, desc="输出张量", title="张量")
    ]

    @staticmethod
    def run(coef, clip, a, b):
        assert isinstance(coef, float), "coef must be a float"

        if clip:
            if coef > 1:
                coef = 1.
            elif coef < 0:
                coef = 0.

        return a * (1 - coef) + b * coef


class TensorIndex(CNodeType):
    """Tensor index.

    Inputs:
        a (Tensor): Input tensor.
        dim (int): axis to index.
        index (int): index.

    Outputs:
        tensor (Tensor): Output tensor.
    """
    TITLE: str = "张量索引"
    PATH: str = "工具"
    DESC: str = "索引张量的特定轴。"
    INPUT_SLOTS = [
        PortConf("a", DataType.TENSOR, 1, desc="输入张量", title="张量"),
        PortConf("dim", DataType.INT, 1, desc="轴", title="轴", default=-1),
        PortConf("index", DataType.INT, 1, desc="索引编号", title="索引编号", default=0)
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, desc="输出张量", title="张量")
    ]

    @staticmethod
    def run(a, dim, index):
        indices = [slice(None)] * a.ndim
        indices[dim] = index
        return a[tuple(indices)]

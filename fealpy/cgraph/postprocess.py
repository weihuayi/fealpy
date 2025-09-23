from .nodetype import CNodeType, PortConf, DataType

__all__ = ["VPDecoupling"]

class VPDecoupling(CNodeType):
    TITLE: str = "速度-压力解耦"
    PATH: str = "后处理.解耦"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解")
    ]

    @staticmethod
    def run(out, uspace):
        ugdof = uspace.number_of_global_dofs()
        uh = out[:ugdof]
        ph = out[ugdof:]

        return uh ,ph

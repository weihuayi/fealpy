
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["MGTensorWPR"]

class MGTensorWPR(CNodeType):
    r"""张量网格上通过多重网格方法求解Stokes方程.
    """
    TITLE: str = "二维水净化反应器模型"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="底面网格"),
        PortConf("imesh", DataType.MESH, 1, title="区间网格"),
        PortConf("thickness", DataType.FLOAT, 1, title="厚度", default=0.4),
        PortConf("smoothing_times", DataType.INT, 1, title="平滑次数", default=1),
        PortConf("tol", DataType.FLOAT, 1, title="容忍误差", default=1e-8),
        PortConf("x0", DataType.TENSOR, 1, title="初始向量", default=None)
    ]
    OUTPUT_SLOTS = [
        PortConf("op", DataType.LINOPS, title="初始系数矩阵"),
        PortConf("F", DataType.TENSOR, title="右端向量"),
        PortConf("Ai", DataType.LINOPS, title="每层的动量方程刚度矩阵"),
        PortConf("Bi", DataType.LINOPS, title="每层的散度矩阵"),
        PortConf("Bti", DataType.LINOPS, title="每层的梯度矩阵"),
        PortConf("bigAi", DataType.TENSOR, title="最粗网格的整体矩阵"),
        PortConf("P_u", DataType.LINOPS, title="速度插值矩阵"),
        PortConf("R_u", DataType.LINOPS, title="速度限制矩阵"),
        PortConf("P_p", DataType.LINOPS, title="压力插值矩阵"),
        PortConf("R_p", DataType.LINOPS, title="压力限制矩阵"),
        PortConf("Nu", DataType.INT, title="每层速度自由度"),
        PortConf("Np", DataType.INT, title="每层压力自由度"),
        PortConf("level", DataType.INT, title="多重网格层数"),
        PortConf("auxMat", DataType.MENU, title="每层LSC-DGS平滑所需若干参数"),
        PortConf("options", DataType.MENU, title="多重网格所需若干参数"),
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.fem import WPRLFEMModel

        mesh = options['mesh']
        imesh = options['imesh']
        model = WPRLFEMModel(options)
        model.set_init_mesher(mesh, imesh)
        model.set_inlet_condition()
        op0, A, F = model.linear_system()
        op, F1, BdDof = model.apply_bc(op0, bm.copy(F))
        del op0
        bd_flag = bm.zeros((len(F),), dtype=bm.bool)
        bm.set_at(bd_flag, BdDof, True)
        model.setup(op)

        return (
            op, F1[~bd_flag], model.Ai, model.Bi, model.Bti,
            model.bigAi, model.P_u, model.R_u, model.P_p, model.R_p,
            model.Nu, model.Np, model.level,
            model.auxMat, model.options
        )


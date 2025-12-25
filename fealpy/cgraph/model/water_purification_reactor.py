
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["MGTensorWPR"]

class MGTensorWPR(CNodeType):
    r"""张量网格上通过多重网格方法求解Stokes方程.
    """
    TITLE: str = "水净化反应器方程离散 (集总端口)"
    PATH: str = "simulation.discretization"
    DESC: str = "三维水净化反应器节点"
    INPUT_SLOTS = [
        PortConf("tmesh", DataType.MESH, 1, title="底面网格"),
        PortConf("imesh", DataType.MESH, 1, title="区间网格"),
        PortConf("inlet_length", DataType.FLOAT, 1, title="入口长度", default=0.5),
        PortConf("inlet_width", DataType.FLOAT, 1, title="入口宽度", default=0.8),
        PortConf("gap", DataType.FLOAT, 1, title="窄缝长度", default=0.1),
        PortConf("gap_len", DataType.FLOAT, 1, title="窄缝宽度", default=1.0),
        PortConf("thickness", DataType.FLOAT, 1, title="厚度", default=0.4),
        PortConf("smoothing_times", DataType.INT, 1, title="平滑次数", default=1),
        PortConf("tol", DataType.FLOAT, 1, title="容忍误差", default=1e-8),
        PortConf("x0", DataType.TENSOR, 1, title="初始向量", default=None)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="三棱柱网格"),
        PortConf("idx", DataType.TENSOR, title="三棱柱网格节点映射"),
        PortConf("op", DataType.LINOPS, title="初始系数矩阵"),
        PortConf("F1", DataType.TENSOR, title="右端向量"),
        PortConf("bd_flag", DataType.TENSOR, title="边界条件"),
        PortConf("ugdof", DataType.INT, title="初始速度自由度"),
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
        import gc

        tmesh = options['tmesh']
        imesh = options['imesh']
        model = WPRLFEMModel(options)
        model.set_init_mesher(tmesh, imesh)
        model.set_inlet_condition()
        op0, A, F = model.linear_system()

        op, F1, BdDof = model.apply_bc(op0, bm.copy(F))
        del op0
        gc.collect()
        bd_flag = bm.zeros((len(F),), dtype=bm.bool)
        bm.set_at(bd_flag, BdDof, True)
        model.setup(op)

        iNN = model.imesh.number_of_nodes()
        tNN = model.tmesh.number_of_nodes()
        tgdof = model.tmesh.number_of_global_ipoints(p=2)
        igdof = model.imesh.number_of_global_ipoints(p=2)
        gdof = tgdof * igdof
        idx = bm.arange(gdof).reshape(tgdof, -1)[:tNN, :iNN].ravel()

        return (
            model.mesh, idx,
            op, F1, bd_flag, model.ugdof, model.Ai, model.Bi, model.Bti,
            model.bigAi, model.P_u, model.R_u, model.P_p, model.R_p,
            model.Nu, model.Np, model.level,
            model.auxMat, model.options
        )


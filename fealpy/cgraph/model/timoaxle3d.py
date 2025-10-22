from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Timoaxle3d"]


class Timoaxle3d(CNodeType):
    r"""3D Timoshenko Beam Axle Geometry Model.
    
     Inputs:
        beam_para (TENSOR): Beam section parameters, each row as [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row as [Diameter, Length, Count].
        kappa (FLOAT): Shear correction factor. Default 10/9.

    Outputs:
        
    """
    TITLE: str = "列车轮轴几何参数模型"
    PATH: str = "模型.几何参数"
    DESC: str = "该节点用于定义列车轮轴系统的几何结构与材料特性参数"
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 0, desc="梁结构参数数组，每行为 [直径, 长度, 数量]",
                title="梁段参数", param="beam_para"),
        PortConf("axle_para", DataType.TENSOR, 0, desc="轴结构参数数组，每行为 [直径, 长度, 数量]",
            title="轴段参数", param="axle_para"),
        PortConf("shear_factors",DataType.FLOAT, 0, desc="梁剪切变形计算中的修因子，圆截面推荐值为 10/9",
                 title="剪切修正因子", param="kappa", default=10/9, min_val=0.0)    
    ]
    
    OUTPUT_SLOTS = [
        PortConf("init_mesh", DataType.MESH, desc="构建轮轴模型的网格", title="网格生成"),
        PortConf("FSY", DataType.FLOAT, desc="Y 方向剪切修正因子，用于剪切变形修正，圆截面推荐值为 10/9",
                 title="Y 方向剪切修正因子"),
        PortConf("FSZ", DataType.FLOAT, desc="Z 方向剪切修正因子，用于剪切变形修正，圆截面推荐值为 10/9",
                 title="Z 方向剪切修正因子"),
        PortConf("calculate_beam_cross_section", DataType.TENSOR, desc="梁的截面参数，包括截面面积分布 Ax, Ay, Az", 
                 title="截面参数", ),
        PortConf("calculate_beam_inertia", DataType.TENSOR, desc="截面的惯性矩分布 Ix, Iy, Iz", title="截面惯性矩"),
        PortConf("external_load", DataType.TENSOR, desc="返回全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_dof_index", DataType.TENSOR, desc="返回 Dirichlet 自由度索引", title="边界自由度索引")
        
    ]

    @staticmethod
    def run(beam_para=None, axle_para=None, kappa=10/9):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        model = TimobeamAxleData3D( beam_para, axle_para, kappa)
        return tuple(
            getattr(model, name)
            for name in ["init_mesh", "FSY", "FSZ", "calculate_beam_cross_section", "calculate_beam_inertia", "external_load", "dirichlet_dof_index"]
        )
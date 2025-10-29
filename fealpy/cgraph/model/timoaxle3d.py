from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Timoaxle3d"]


class Timoaxle3d(CNodeType):
    r"""3D Timoshenko Beam-Axle Geometry Model.
    
     Inputs:
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        section_shapes (MENU): Beam cross-section shape configuration parameter.
        shear_factors (FLOAT): Shear correction factor. Default 10/9.

    Outputs:
        init_mesh(mesh): Create a mesh object.
        external_load (function): Function that returns the global load vector.
        dirichlet_dof_index (function): Function that returns Dirichlet boundary condition indices.
    """
    TITLE: str = "列车轮轴几何参数模型"
    PATH: str = "模型.几何参数"
    DESC: str = "定义列车轮轴系统的几何结构、剪切修正因子及边界条件函数"
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 0, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 0, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("section_shapes", DataType.MENU, 0, desc="梁的截面形状", title="梁截面形状", default="circular", 
                 items=["circular", "rectangular", "I-shaped", "H-shaped"]),
        PortConf("shear_factors",DataType.FLOAT, 0, desc="梁剪切变形计算中的修因子，圆截面推荐值为 10/9",
                 title="剪切修正因子", param="kappa", default=10/9, min_val=0.0)    
    ]
    
    OUTPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("external_load", DataType.FUNCTION, desc="全局载荷向量的函数", title="外部载荷"),
        PortConf("dirichlet_dof_index", DataType.FUNCTION, desc="Dirichlet 自由度索引的函数", title="边界自由度索引")
        
    ]

    @staticmethod
    def run(beam_para=None, axle_para=None, section_shapes="circular", kappa=None):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        model = TimobeamAxleData3D( beam_para, axle_para, kappa)
        return tuple(
            getattr(model, name)
            for name in ["beam_para", "axle_para", "external_load", "dirichlet_dof_index"]
        )
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
        GD (INT): Geometric dimension of the model.
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        R (TENSOR): Transformation matrix between global and local coordinates.
        shear_factors (FLOAT): Shear correction factor. Default 10/9.
        external_load (function): Function that returns the global load vector.
        dirichlet_dof (function): Function that returns Dirichlet boundary condition indices.
    """
    TITLE: str = "列车轮轴几何参数模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点用于建立列车轮轴的三维几何参数模型，通过梁段与轴段的结构参数、截面形状及剪切修正因子，
            定义 Timoshenko 梁模型所需的关键几何特性。节点负责输出完整的梁/轴分段参数、截面类型、外部载荷函数
            以及边界自由度索引，为后续有限元离散与静力分析提供基础几何信息。"""
            
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 0, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 0, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("section_shapes", DataType.MENU, 0, desc="梁的截面形状", title="梁截面形状", default="circular", 
                 items=["circular", "rectangular", "I-shaped", "H-shaped"]),
        PortConf("shear_factors",DataType.FLOAT, 0, desc="梁剪切变形计算中的修因子，圆截面推荐值为 10/9",
                 title="剪切修正因子", param="kappa", default=10/9)    
    ]
    
    OUTPUT_SLOTS = [
        PortConf("GD", DataType.INT, title="几何维数"),
        PortConf("beam_para", DataType.TENSOR, title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, title="轴段参数"),
        PortConf("coord_transform", DataType.TENSOR, title="坐标变换矩阵"),
        PortConf("external_load", DataType.FUNCTION, title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, title="边界自由度索引")
        
    ]

    @staticmethod
    def run(beam_para=None, axle_para=None, section_shapes="circular", kappa=10/9):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        
        model = TimobeamAxleData3D(beam_para, axle_para)
        
        external_load = model.external_load()
        dirichlet_dof = model.dirichlet_dof()
        coord_transform = model.coord_transform()

        return (model.GD, model.beam_para, model.axle_para, coord_transform,
                external_load, dirichlet_dof)
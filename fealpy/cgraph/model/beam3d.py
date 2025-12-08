from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ChannelBeam3d", "Timoaxle3d"]


class ChannelBeam3d(CNodeType):
    r"""3D Channel Beam Geometry Model.
    
    Inputs:
        mu_y (FLOAT): Ratio of maximum to average shear stress for y-direction shear. Default 2.44.
        mu_z (FLOAT): Ratio of maximum to average shear stress for z-direction shear. Default 2.38.
        n_elements (INT): Number of elements along the beam length. Default 10.
        load_case (INT): Load case number (1 or 2). Default 1.

    Outputs:
        GD (INT): Geometric dimension of the model.
        mesh (MESH): 1D mesh along the beam length.
        mu_y (FLOAT): Ratio of maximum to average shear stress for y-direction shear. Default 2.44.
        mu_z (FLOAT): Ratio of maximum to average shear stress for z-direction shear. Default 2.38.
        load_case (INT): Load case number (1 or 2). Default 1.
        dirichlet_dof (FUNCTION): Function that returns Dirichlet boundary condition indices.
    """
    TITLE: str = "槽形梁几何参数模型"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("mu_y", DataType.FLOAT, 0, desc="y方向剪切应力的最大值与平均值比例因子", 
                 title="y向剪切因子", default=2.44),
        PortConf("mu_z", DataType.FLOAT, 0, desc="z方向剪切应力的最大值与平均值比例因子", 
                 title="z向剪切因子", default=2.38),
        PortConf("load_case", DataType.MENU, 0, desc="载荷工况选择", 
                 title="载荷工况", default=1, items=[1, 2])
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mu_y", DataType.FLOAT, title="y向剪切因子"),
        PortConf("mu_z", DataType.FLOAT, title="z向剪切因子"),
        PortConf("GD", DataType.INT, title="几何维数"),
        PortConf("load_case", DataType.MENU, title="载荷工况"),
        PortConf("dirichlet_dof", DataType.FUNCTION, title="边界自由度")
    ]

    @staticmethod
    def run(**options):
        from fealpy.csm.model.beam.channel_beam_data_3d import ChannelBeamData3D
        
        mu_y = options.get("mu_y")
        mu_z = options.get("mu_z")
        load_case = options.get("load_case")
        
        model = ChannelBeamData3D(mu_y=mu_y, mu_z=mu_z)
        
        load_case = load_case
        dirichlet_dof = model.dirichlet_dof()

        return (mu_y, mu_z, model.GD, 
                load_case, dirichlet_dof)
        

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
        shear_factors (FLOAT): Shear correction factor. Default 10/9.
        external_load (function): Function that returns the global load vector.
        dirichlet_dof (function): Function that returns Dirichlet boundary condition indices.
    """
    TITLE: str = "列车车轴几何参数模型"
    PATH: str = "preprocess.modeling"
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
        PortConf("external_load", DataType.FUNCTION, title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, title="边界自由度索引")
        
    ]

    @staticmethod
    def run(beam_para=None, axle_para=None, section_shapes="circular", kappa=10/9):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        
        model = TimobeamAxleData3D(beam_para, axle_para)
        
        external_load = model.external_load()
        dirichlet_dof = model.dirichlet_dof()

        return (model.GD, model.beam_para, model.axle_para,
                external_load, dirichlet_dof)
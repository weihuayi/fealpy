from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["TimoMaterial", "AxleMaterial"]


class TimoMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.
    
        Inputs:
            property (string): Material type, e.g., "Steel".
            beam_type (menu): Beam model type selection.
            beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
            axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
            beam_E (float): Elastic modulus of the beam material.
            beam_nu (float): Poisson’s ratio of the beam material.
            
        Outputs:
            property (string): Material type.
            beam_type (menu): Beam model type.
            E (float): Elastic modulus of the beam material.
            mu (float): Shear modulus, computed as `E / [2(1 + nu)]`.
    """
    TITLE: str = "列车轮轴梁材料属性"
    PATH: str = "preprocess.material"
    DESC: str = """该节点用于定义列车轮轴中梁段（Beam）部分的材料属性，
        并根据输入的梁几何参数和材料参数计算材料的基本力学常数，
        包括弹性模量、泊松比和剪切模量。节点同时支持铁木辛柯梁模型。"""
        
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="轮轴材料类型选择", title="梁材料", default="Timo_beam", 
                 items=["Euler_beam", "Timo_beam"]),
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁的弹性模量", default=2.1e11),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁的泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="梁的泊松比"),
        PortConf("mu", DataType.FLOAT, title="梁的剪切模量")
    ]
    
    @staticmethod
    def run(property="Steel", beam_type="Timoshemko_beam", 
            beam_para=None, axle_para=None,
            beam_E=2.1e11, beam_nu=0.3):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        
        model = TimobeamAxleData3D(beam_para, axle_para)
        
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name=beam_type,
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)
        
        return tuple(
            getattr(beam_material, name)
            for name in ["E", "nu", "mu"]
        )
        

class AxleMaterial(CNodeType):
    r"""Axle Material Definition Node.
    
        Inputs:
            property (string): Material name, e.g., "Steel".
            axle_type (menu): Type of axle material.
            beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
            axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
            axle_stiffness (float): spring stiffness.
            axle_E (float): Elastic modulus of the axle material.
            axle_nu (float): Poisson’s ratio of the axle material.
 
        Outputs:
            E (float): Elastic modulus of the axle material.
            nu (float): Poisson’s ratio of the axle material.
            mu (float): Shear modulus, computed as `E / [2(1 + nu)]`.
    
    """
    TITLE: str = "列车轮轴弹簧材料属性"
    PATH: str = "preprocess.material"
    DESC: str = """该节点计算列车轮轴系统中杆件（Spring / Axle-Rod）部分的材料属性，
        包括弹性模量、泊松比和剪切模量。适用于需要对轴上的弹簧、连接杆等结构进行建模的场景。"""
        
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("axle_type", DataType.STRING, 0, desc="轮轴材料类型", title="弹簧材料", default="Spring"),
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("axle_stiffness", DataType.FLOAT, 0, desc="弹簧刚度", title="弹簧的刚度", default=1.976e6),
        PortConf("axle_E", DataType.FLOAT, 0, desc="弹簧弹性模量", title="弹簧的弹性模量", default=1.976e6),
        PortConf("axle_nu", DataType.FLOAT, 0, desc="弹簧泊松比", title="弹簧的泊松比", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹簧的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="弹簧的泊松比")
    ]
        
    @staticmethod
    def run(property, axle_type, axle_stiffness, 
            beam_para=None, axle_para=None, 
            axle_E=1.976e6, axle_nu=-0.5):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import BarMaterial
        
        model = TimobeamAxleData3D(beam_para, axle_para)
        
        axle_material = BarMaterial(model=model,
                                name=axle_type,
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)
        return tuple(
            getattr(axle_material, name) for name in ["E", "nu"])
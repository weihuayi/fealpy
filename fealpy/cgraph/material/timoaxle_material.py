from typing import Union
from ..nodetype import CNodeType, PortConf, DataType


class TimoaxleMaterial(CNodeType):
    r"""Timoshenko Beam and Axle Material Definition Node.
    
        Inputs:
            shear_factor (FLOAT): Shear correction factor, obtained from geometry node.
            
        Outputs:
        E (FLOAT): Elastic modulus
        nu (FLOAT): Poisson's ratio
        mu (FLOAT): Shear modulus
    
    """
    TITLE: str = "列车轮轴材料定义"
    PATH: str = "模型.材料定义"
    DESC: str = "定义列车轮轴系统的材料属性与模型常数"
    INPUT_SLOTS = [
        PortConf("material_property", DataType.STRING, 0, desc="材料材质",
            title="材料材质", param="property", default="Steel"),
        PortConf("beam_material", DataType.MENU, 0, desc="梁材料类型选择", title="梁材料",
                 items=["Timoshenko", "Euler-Bernoulli"]),
        PortConf("axle_material", DataType.MENU, 0, desc="轮轴材料类型选择", title="轮轴材料",
                 items=["Bar", "other"]),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁弹性模量", default=2.1e11),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁泊松比", default=0.3),
        PortConf("axle_E", DataType.FLOAT, 0, desc="轮轴的弹性模量", title="轮轴弹性模量", default=1.976e6),
        PortConf("axle_nu", DataType.FLOAT, 0, desc="轮轴的泊松比", title="轮轴泊松比", param="axle_nu", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("beam_material", DataType.FUNCTION, desc="包含梁的弹性模量、泊松比、剪切模量等",  title="梁的材料性质"),
        PortConf("axle_material", DataType.FUNCTION, desc="包含轮轴的弹性模量、泊松比、剪切模量等", title="轮轴的材料性质")
    ]
    
    @staticmethod
    def run(beam_E, beam_nu, axle_E, axle_nu):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import AxleMaterial, TimoshenkoBeamMaterial
        model = TimobeamAxleData3D()
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="timobeam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)
        axle_material = AxleMaterial(model=model,
                                name="axle",
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)
        return beam_material, axle_material
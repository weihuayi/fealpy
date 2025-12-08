from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarMaterial", 
           "AxleMaterial"]

 
class BarMaterial(CNodeType):
    r"""A CGraph node for defining the material properties of a 3D truss/bar.

    Inputs:
        property(MENU): Material name, e.g."structural-steel".
        bar_type(STRING): Type of bar material.
        E(FLOAT): Elastic modulus of the bar material.
        nu(FLOAT): Poisson's ratio of the bar material.

    Outputs:
        E (float): The elastic modulus of the bar material.
        nu(FLOAT): Poisson's ratio of the bar material.

    """
    TITLE: str = "杆单元材料属性"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("bar_type", DataType.MENU, 0, desc="杆件结构类型", title="杆件类型", default="custom",
                 items=["bar25", "bar942", "truss_tower", "custom"]),
        PortConf("E", DataType.FLOAT, 0, desc="杆的弹性模量", title="弹性模量", default=2.0e11),
        PortConf("nu", DataType.FLOAT, 0, desc="杆的泊松比", title="泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹性模量"),
        PortConf("nu", DataType.FLOAT, title="泊松比")
    ]
    
    @staticmethod
    def run(**options):
        bar_type = options.get("bar_type")
        E = options.get("E")
        nu = options.get("nu")
        
        # 定义各类型的预设材料参数
        presets = {
            "bar25": {"E": 1500, "nu": 0.3},
            "bar942": {"E": 2.1e5, "nu": 0.3},
            "truss_tower": {"E": 2.0e11, "nu": 0.3},
            "custom": {"E": E, "nu": nu}  # custom使用用户输入
        }

        preset = presets.get(bar_type, {})
        E = preset.get("E", E)
        nu = preset.get("nu", nu)
        return E, nu

class AxleMaterial(CNodeType):
    r"""Axle Material Definition Node.
    
    Inputs:
        property(MENU): Material name, e.g."structural-steel".
        type(STRING): Type of bar material.
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        stiffness (FLOAT): spring stiffness.
        E (FLOAT): Elastic modulus of the axle material.
        nu (FLOAT): Poisson's ratio of the axle material.

        Outputs:
            E (FLOAT): Elastic modulus of the axle material.
            nu (FLOAT): Poisson's ratio of the axle material.
            mu (FLOAT): Shear modulus of the axle material.

    """
    TITLE: str = "列车车轴弹簧部分材料属性"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.STRING, 0, desc="材料类型", title="弹簧类型", default="spring"),
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("stiffness", DataType.FLOAT, 0, desc="弹簧刚度", title="弹簧的刚度", default=1.976e6),
        PortConf("E", DataType.FLOAT, 0, desc="弹簧弹性模量", title="弹簧的弹性模量", default=1.976e6),
        PortConf("nu", DataType.FLOAT, 0, desc="弹簧泊松比", title="弹簧的泊松比", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹簧的弹性模量"),
         PortConf("nu", DataType.FLOAT, title="弹簧的泊松比"),
        PortConf("mu", DataType.FLOAT, title="弹簧的剪切模量")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import AxleMaterial

        model = TimobeamAxleData3D(
            beam_para=options.get("beam_para"),
            axle_para=options.get("axle_para")
        )

        axle_material = AxleMaterial(model=model,
                                name=options.get("type"),
                                elastic_modulus=options.get("E"),
                                poisson_ratio=options.get("nu")
                            )
        return tuple(
            getattr(axle_material, name) for name in ["E", "nu", "mu"]
        )
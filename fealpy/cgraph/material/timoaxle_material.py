from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["TimoMaterial", "AxleMaterial"]


class TimoMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.
    
        Inputs:
            shear_factor (FLOAT): Shear correction factor, obtained from geometry node.
            
        Outputs:
        E (FLOAT): Elastic modulus
        nu (FLOAT): Poisson's ratio
        mu (FLOAT): Shear modulus
    
    """
    TITLE: str = "列车轮轴材料定义"
    PATH: str = "模型.材料"
    DESC: str = "定义列车轮轴系统的梁材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料材质", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="梁材料类型选择", title="梁材料类型",
                 items=["Timoshenko", "Euler-Bernoulli"]),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁弹性模量", default=2.1e11),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("property", DataType.STRING, desc="材料材质", title="材料材质"),
        PortConf("beam_type", DataType.MENU, desc="梁材料类型", title="材料类型"),
        PortConf("E", DataType.FLOAT, desc="弹性模量",  title="梁的材料属性"),
        PortConf("mu", DataType.FLOAT, desc="剪切模量",  title="梁的材料属性"),
        PortConf("Ax", DataType.FLOAT, desc="X 轴横截面积",  title="横截面积"),
        PortConf("Ay", DataType.FLOAT, desc="Y 轴的横截面积",  title="横截面积"),
        PortConf("Az", DataType.FLOAT, desc="Z 轴的横截面积",  title="横截面积"),
        PortConf("J", DataType.FLOAT, desc="X 轴的惯性矩",  title="极性矩"),
        PortConf("Iy", DataType.FLOAT, desc="Y 轴的惯性矩",  title="惯性矩"),
        PortConf("Iz", DataType.FLOAT, desc="Z 轴的惯性矩",  title="惯性矩")
    ]
    
    @staticmethod
    def run(property="Steel", beam_type="Timoshemko beam", beam_E=2.1e11, beam_nu=0.3):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        model = TimobeamAxleData3D()
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="timobeam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)
        
        return (property, beam_type) + tuple(
            getattr(beam_material, name)
            for name in ["E", "mu", "Ax", "Ay", "Az", "J", "Iy", "Iz"]
        )
        

class AxleMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.
    
        Inputs:
            shear_factor (FLOAT): Shear correction factor, obtained from geometry node.
            
        Outputs:
        E (FLOAT): Elastic modulus
        nu (FLOAT): Poisson's ratio
        mu (FLOAT): Shear modulus
    
    """
    TITLE: str = "列车轮轴材料定义"
    PATH: str = "模型.材料定义"
    DESC: str = "定义列车轮轴系统的杆材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料材质", title="材料材质", default="Steel"),
        PortConf("axle_type", DataType.MENU, 0, desc="轮轴材料类型选择", title="轮轴材料类型", items=["Bar", "other"]),
        PortConf("axle_E", DataType.FLOAT, 0, desc="杆的弹性模量", title="弹性模量", default=1.976e6),
        PortConf("axle_nu", DataType.FLOAT, 0, desc="杆的泊松比", title="泊松比", param="axle_nu", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("property", DataType.STRING, desc="材料材质", title="材料材质"),
        PortConf("axle_type", DataType.MENU, desc="轮轴材料类型", title="轮轴材料类型"),
        PortConf("E", DataType.FLOAT, desc="弹性模量",  title="杆的材料属性"),
        PortConf("mu", DataType.FLOAT, desc="剪切模量",  title="杆的材料属性")
    ]
        
    @staticmethod
    def run(property="Steel", axle_type="Bar",axle_E=1.97e6, axle_nu=-0.5):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import AxleMaterial
        model = TimobeamAxleData3D()
        axle_material = AxleMaterial(model=model,
                                name="axle",
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)
        return (property, axle_type) + tuple(
            getattr(axle_material, name) for name in ["E", "mu"])
from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BeamMaterial", 
           "ChannelBeamMaterial",
           "TimoMaterial"]


class BeamMaterial(CNodeType):
    r"""Euler-Bernoulli Beam Material Definition Node.
    
        Inputs:
            property (string): Material type (e.g., Steel, Aluminum).
            beam_type (menu): Beam model type selection.
            beam_E (float): Elastic modulus of the beam.
            beam_nu (float): Poisson's ratio of the beam.
            I (float): Second moment of area (area moment of inertia).
            
        Outputs:
            I (tensor): Second moment of area.
            E (float): Elastic modulus of the beam.
            nu (float): Poisson's ratio of the beam.
    """
    TITLE: str = "欧拉梁材料属性"
    PATH: str = "material.solid"
    DESC: str = "欧拉梁材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="梁模型类型选择", title="梁材料类型", default="Euler-Bernoulli",
                 items=["Euler-Bernoulli", "Timoshenko"]),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁弹性模量", default=200e9),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁泊松比", default=0.3),
        PortConf("I", DataType.FLOAT, 0, desc="惯性矩", title="惯性矩", default=118.6e-6),
        
    ]
    
    OUTPUT_SLOTS = [
        PortConf("I", DataType.TENSOR, title="惯性矩"),
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="梁的泊松比"),
    ]
    
    @staticmethod
    def run(property="Steel", beam_type="Euler-Bernoulli beam", beam_E=200e9, beam_nu=0.3, I=118.6e-6):
        from fealpy.csm.model.beam.euler_bernoulli_beam_data_2d import EulerBernoulliBeamData2D
        from fealpy.csm.material import EulerBernoulliBeamMaterial
        model = EulerBernoulliBeamData2D()
        beam_material = EulerBernoulliBeamMaterial(model=model, 
                                        name="eulerbeam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu,
                                        I=I)
        
        return (beam_material.I,)+tuple(
            getattr(beam_material, name)
            for name in ["E", "nu"]
        )


class ChannelBeamMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.

    Inputs:
        property (STRING): Material type, e.g., "Steel".
        beam_type (MENU): Beam model type selection.
        mu_y (FLOAT): Ratio of maximum to average shear stress for y-direction shear.
        mu_z (FLOAT): Ratio of maximum to average shear stress for z-direction shear.
        E (FLOAT): Elastic modulus of the beam material.
        nu (FLOAT): Poisson's ratio of the beam material.
        density (FLOAT): Density of the beam material.

    Outputs:
        property (STRING): Material type.
        beam_type (MENU): Beam model type.
        E (FLOAT): Elastic modulus of the beam material.
        mu (FLOAT): Shear modulus, computed as `E / [2(1 + nu)]`.
    """
    TITLE: str = "槽形梁材料属性"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.MENU, 0, desc="材料类型选择", title="梁类型", default="Timoshenko", 
                items=["Euler-Bernoulli", "Timoshenko"]),
        PortConf("mu_y", DataType.FLOAT, 1, desc="y方向剪切应力比", title="y方向剪切应力比", default=2.44),
        PortConf("mu_z", DataType.FLOAT, 1, desc="z方向剪切应力比", title="z方向剪切应力比", default=2.38),
        PortConf("E", DataType.FLOAT, 0, desc="梁的弹性模量", title="弹性模量", default=2.1e11),
        PortConf("nu", DataType.FLOAT, 0, desc="梁的泊松比", title="泊松比", default=0.25),
        PortConf("density", DataType.FLOAT, 0, desc="梁的密度", title="密度", default=7800)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="梁的泊松比"),
        PortConf("mu", DataType.FLOAT, title="梁的剪切模量"),
        PortConf("rho", DataType.FLOAT, title="梁的密度")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.beam.channel_beam_data_3d import ChannelBeamData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        
        mu_y = options.get("mu_y")
        mu_z = options.get("mu_z")
        model = ChannelBeamData3D(mu_y=mu_y, mu_z=mu_z)

        beam_material = TimoshenkoBeamMaterial(model=model,
                                        name=options.get("type"),
                                        elastic_modulus=options.get("E"),
                                        poisson_ratio=options.get("nu"),
                                        density=options.get("density"))

        return tuple(
            getattr(beam_material, name)
            for name in ["E", "nu", "mu", "rho"]
        )
        

class TimoMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.
    
        Inputs:
            property (STRING): Material type, e.g., "Steel".
            beam_type (MENU): Beam model type selection.
            beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
            axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
            E (FLOAT): Elastic modulus of the beam material.
            nu (FLOAT): Poisson's ratio of the beam material.

        Outputs:
            E (FLOAT): Elastic modulus of the beam material.
            nu (FLOAT): Poisson's ratio of the beam material.
            mu (FLOAT): Shear modulus, computed as `E / [2(1 + nu)]`.
    """
    TITLE: str = "列车车轴梁段部分材料属性"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.MENU, 0, desc="车轴材料类型选择", title="梁类型", default="Timoshenko", 
                 items=["Euler-Bernoulli", "Timoshenko"]),
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁的弹性模量", default=2.1e11),
        PortConf("nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁的泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="梁的泊松比"),
        PortConf("mu", DataType.FLOAT, title="梁的剪切模量")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial

        model = TimobeamAxleData3D(
            beam_para=options.get("beam_para"),
            axle_para=options.get("axle_para")
        )

        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name=options.get("type"),
                                        elastic_modulus=options.get("E"),
                                        poisson_ratio=options.get("nu"))

        return tuple(
            getattr(beam_material, name)
            for name in ["E", "nu", "mu"]
        )
        
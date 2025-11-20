from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BeamMaterial"]


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
    PATH: str = "材料.固体"
    DESC: str = "欧拉梁材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="梁模型类型选择", title="梁材料类型",
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
        
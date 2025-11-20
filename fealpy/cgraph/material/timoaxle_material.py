from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["TimoMaterial", "AxleMaterial"]


class TimoMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.
    
        Inputs:
            property (string): Material type, e.g., "Steel".
            beam_type (menu): Beam model type selection.
            beam_E (float): Elastic modulus of the beam material.
            beam_nu (float): Poisson’s ratio of the beam material.
            
        Outputs:
            property (string): Material type.
            beam_type (menu): Beam model type.
            E (float): Elastic modulus of the beam material.
            mu (float): Shear modulus, computed as `E / [2(1 + nu)]`.
            Ax (float): Cross-sectional area in the X direction.
            Ay (float): Cross-sectional area in the Y direction.
            Az (float): Cross-sectional area in the Z direction.
            J (float): Polar moment of inertia about the X axis.
            Iy (float): Moment of inertia about the Y axis.
            Iz (float): Moment of inertia about the Z axis.
    """
    TITLE: str = "列车轮轴梁材料属性"
    PATH: str = "材料.固体"
    DESC: str = "定义列车轮轴系统的梁材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="轮轴材料类型选择", title="梁材料", default="Timo_beam", 
                 items=["Euler_beam", "Timo_beam"]),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁弹性模量", default=2.1e11),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("mu", DataType.FLOAT, title="梁的剪切模量"),
        PortConf("Ax", DataType.TENSOR, title="X 方向横截面积"),
        PortConf("Ay", DataType.TENSOR, title="Y 方向横截面积"),
        PortConf("Az", DataType.TENSOR, title="Z 方向横截面积"),
        PortConf("J", DataType.TENSOR,  title="X 轴极惯性矩（扭转惯性矩）"),
        PortConf("Iy", DataType.TENSOR, title="Y 轴惯性矩"),
        PortConf("Iz", DataType.TENSOR, title="Z 轴惯性矩")
    ]
    
    @staticmethod
    def run(property="Steel", beam_type="Timoshemko beam", beam_E=None, beam_nu=None):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        model = TimobeamAxleData3D()
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="timobeam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)
        
        return tuple(
            getattr(beam_material, name)
            for name in ["E", "mu", "Ax", "Ay", "Az", "J", "Iy", "Iz"]
        )
        

class AxleMaterial(CNodeType):
    r"""Axle Material Definition Node.
    
        Inputs:
            property (string): Material name, e.g., "Steel".
            axle_type (menu): Type of axle material.
            axle_stiffness (float): spring stiffness.
            axle_E (float): Elastic modulus of the axle material.
            axle_nu (float): Poisson’s ratio of the axle material.
 
        Outputs:
            property (string): Material name.
            axle_type (menu): Axle material type.
            E (float): Elastic modulus of the axle material.
            mu (float): Shear modulus, computed as `E / [2(1 + nu)]`.
    
    """
    TITLE: str = "列车轮轴弹簧材料属性"
    PATH: str = "材料.固体"
    DESC: str = "定义列车轮轴系统中弹簧的材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("axle_type", DataType.STRING, 0, desc="轮轴材料类型", title="弹簧材料", default="Spring"),
        PortConf("axle_stiffness", DataType.FLOAT, 0, desc="弹簧刚度", title="弹簧的刚度", default=1.976e6),
        PortConf("axle_E", DataType.FLOAT, 0, desc="弹性模量", title="弹簧的弹性模量", default=1.976e6),
        PortConf("axle_nu", DataType.FLOAT, 0, desc="泊松比", title="弹簧的泊松比", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹簧的弹性模量"),
        PortConf("mu", DataType.FLOAT, title="弹簧的剪切模量")
    ]
        
    @staticmethod
    def run(property, axle_type, axle_stiffness, axle_E, axle_nu):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import AxleMaterial
        model = TimobeamAxleData3D()
        axle_material = AxleMaterial(model=model,
                                name="axle",
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)
        return tuple(
            getattr(axle_material, name) for name in ["E", "mu"])
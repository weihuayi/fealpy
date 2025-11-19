from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["TimoMaterial", "AxleMaterial", "TimoAxleStrainStress"]


class TimoMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.
    
        Inputs:
            property (STRING): Material type, e.g., "Steel".
            beam_type (MENU): Beam model type selection.
            beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
            axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
            beam_E (FLOAT): Elastic modulus of the beam material.
            beam_nu (FLOAT): Poisson’s ratio of the beam material.

        Outputs:
            property (STRING): Material type.
            beam_type (MENU): Beam model type.
            E (FLOAT): Elastic modulus of the beam material.
            mu (FLOAT): Shear modulus, computed as `E / [2(1 + nu)]`.
    """
    TITLE: str = "列车轮轴梁材料属性"
    PATH: str = "material.solid"
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
        property (STRING): Material name, e.g., "Steel".
        axle_type (MENU): Type of axle material.
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        axle_stiffness (FLOAT): spring stiffness.
        axle_E (FLOAT): Elastic modulus of the axle material.
        axle_nu (FLOAT): Poisson’s ratio of the axle material.

        Outputs:
            E (FLOAT): Elastic modulus of the axle material.
            nu (FLOAT): Poisson’s ratio of the axle material.

    """
    TITLE: str = "列车轮轴弹簧材料属性"
    PATH: str = "material.solid"
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
        

class TimoAxleStrainStress(CNodeType):
    r"""compute Strain and Stress for train-axle model.
    
        Inputs:
            beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
            axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
            beam_E (FLOAT): Elastic modulus of the beam material.
            beam_nu (FLOAT): Poisson’s ratio of the beam material.
            axle_E (FLOAT): Elastic modulus of the axle material.
            axle_nu (FLOAT): Poisson’s ratio of the axle material.
            mesh (MESH): Mesh containing node and cell information.
            uh (TENSOR): Post-processed displacement vector.
            y (FLOAT): Local coordinates in the beam cross-section.
            z (FLOAT): Local coordinates in the beam cross-section.
            axial_position (FLOAT)): Evaluation position along the beam axis ∈ [0, L].
                If None, the value is evaluated at the element midpoint L/2
            beam_num (INT): Number of beam elements. If None, uses all cells.
            axle_num (INT): Number of axle elements. If None, uses all cells.

        Outputs:
            beam_strain (TENSOR): Strain of the beam elements.
            beam_stress (TENSOR): Stress of the beam elements.
            axle_strain (TENSOR): Strain of the axle elements.
            axle_stress (TENSOR): Stress of the axle elements.
            strain (TENSOR): Strain of the bar elements.
            stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "列车轮轴应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点基于线弹性理论，对列车轮轴结构的杆件执行应变–应力计算。
            节点通过单元网格、材料参数以及位移场，计算相应的单元应变及应力，用于结构后处理与安全性分析。
            并且用户可选择特定单元进行计算，或对所有单元执行统一的应变–应力分析。"""
            
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁段的弹性模量", title="梁的弹性模量"),
        PortConf("beam_nu", DataType.FLOAT, 1, desc="梁段的泊松比", title="梁的泊松比"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="轴段的弹性模量", title="弹簧的弹性模量"),
        PortConf("axle_nu", DataType.FLOAT, 1, desc="轴段的泊松比", title="弹簧的泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移向量", title="位移向量"),
        PortConf("y", DataType.FLOAT, 0, desc="截面局部 Y 坐标", title="Y坐标", default=0.0),
        PortConf("z", DataType.FLOAT, 0, desc="截面局部 Z 坐标", title="Z坐标", default=0.0),
        PortConf("axial_position", DataType.INT, 0, desc="轴向评估位置", title="轴向位置", default=None),
        PortConf("beam_num", DataType.INT, 0, desc="梁单元个数", title="梁单元", default=23),
        PortConf("axle_num", DataType.INT, 0, desc="弹簧单元个数", title="弹簧单元", default=10),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("beam_strain", DataType.TENSOR, title="梁应变"),
        PortConf("beam_stress", DataType.TENSOR, title="梁应力"),
        PortConf("axle_strain", DataType.TENSOR, title="轴应变"),
        PortConf("axle_stress", DataType.TENSOR, title="轴应力"),
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        from fealpy.csm.material import BarMaterial
        
        mesh = options.get("mesh")
        NC = mesh.number_of_cells()
        
        uh = options.get("uh").reshape(-1, 6)
        
        model = TimobeamAxleData3D(
                beam_para=options.get("beam_para"),
                axle_para=options.get("axle_para")
            )
        
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timo_beam",
                                        elastic_modulus=options.get("beam_E"),
                                        poisson_ratio=options.get("beam_nu"))

        beam_num = options.get("beam_num")
        beam_indices = bm.arange(0, beam_num)

        R = model.coord_transform(index=beam_indices)
        
        y = options.get("y")
        z = options.get("z")
        axial_position = options.get("axial_position")
        
        beam_strain, beam_stress = beam_material.compute_strain_and_stress(
                        mesh,
                        uh,
                        cross_section_coords=(y, z),
                        axial_position=axial_position,
                        coord_transform=R,
                        ele_indices=beam_indices)

        axle_num = options.get("axle_num")
        axle_indices = bm.arange(NC-axle_num, NC)

        axle_material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("axle_E"),
            poisson_ratio=options.get("axle_nu")
        )
        
        axle_strain, axle_stress = axle_material.compute_strain_and_stress(
                        mesh,
                        uh,
                        ele_indices=axle_indices)
        
        strain = bm.zeros((NC, 3), dtype=bm.float64)
        stress = bm.zeros((NC, 3), dtype=bm.float64)
        
        strain[beam_indices] = beam_strain
        stress[beam_indices] = beam_stress
        
        strain[axle_indices] = axle_strain
        stress[axle_indices] = axle_stress
        
        return (beam_strain, beam_stress,
                axle_strain, axle_stress, 
                strain, stress, )
from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarMaterial", "TrussTowerMaterial", "BarStrainStress"]


class BarMaterial(CNodeType):
    r"""
    A CGraph node for defining the material properties of a 3D truss/bar.

    This node takes basic engineering properties for a bar element, such as elastic modulus and 
    cross-sectional area, and provides them as outputs for use in downstream finite element
    analysis nodes.

    Inputs:
        property (string): The name of the material (e.g., "Steel").
        bar_E (float): The elastic modulus (Young's modulus) of the bar material.
        bar_A (float): The cross-sectional area of the bar.

    Outputs:
        E (float): The elastic modulus of the bar material.
        A (float): The cross-sectional area of the bar.
    """
    TITLE: str = "桁架杆件材料属性"
    PATH: str = "材料.固体"
    DESC: str = "定义桁架杆件的线弹性材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("bar_E", DataType.FLOAT, 0, desc="杆的弹性模量", title="杆弹性模量", default=1500),
        PortConf("bar_A", DataType.FLOAT, 0, desc="杆的横截面积", title="杆横截面积", default=2000),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="杆的弹性模量"),
        PortConf("A", DataType.FLOAT, title="杆的横截面积"),
    ]
    
    @staticmethod
    def run(property="Steel", bar_E=1500, bar_A=2000):
        return bar_E, bar_A
    
    
class TrussTowerMaterial(CNodeType):
    r"""Axle Material Definition Node.
    
        Inputs:
            property (string): Material name, e.g., "Steel".
            type (string): Type of axle material.
            dov (float): Outer diameter of vertical rods (m).
            div (float): Inner diameter of vertical rods (m).
            doo (float): Outer diameter of other rods (m).
            dio (float): Inner diameter of other rods (m).
            E (float): Elastic modulus of the axle material.
            nu (float): Poisson’s ratio of the axle material.
 
        Outputs:
            E (float): Elastic modulus of the axle material.
            mu (float): Shear modulus, computed as `E / [2(1 + nu)]`.
    
    """
    TITLE: str = "桁架塔材料属性"
    PATH: str = "material.solid"
    DESC: str = """该节点定义桁架塔结构中所使用的杆件材料参数，包括材料名称、类型、弹性模量与泊松比等内容，
            通过标准化的材料属性输出，为有限元单元刚度计算与结构分析提供统一的材料基础。"""
            
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("type", DataType.STRING, 0, desc="材料类型", title="杆件材料", default="bar"),
        PortConf("dov", DataType.FLOAT, 1,  desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, 1,  desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, 1,  desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, 1,  desc="其他杆件的内径", title="其他杆内径"),
        PortConf("E", DataType.FLOAT, 0, desc="杆件的弹性模量", title="弹性模量", default=2.0e11),
        PortConf("nu", DataType.FLOAT, 0, desc="杆件的泊松比", title="泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹性模量"),
        PortConf("nu", DataType.FLOAT, title="泊松比"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        
        model = TrussTowerData3D(
            dov=options.get("dov"),
            div=options.get("div"),
            doo=options.get("doo"),
            dio=options.get("dio")
        )
        material = BarMaterial(
            model=model,
            name=options.get("type"),
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )

        return tuple(
            getattr(material, name) for name in ["E", "nu"]
        )
        

class BarStrainStress(CNodeType):
    r"""compute Strain and Stress for Bar Elements.
    
    Inputs:
        dov (FLOAT): Outer diameter of vertical rods (m).
        div (FLOAT): Inner diameter of vertical rods (m).
        doo (FLOAT): Outer diameter of other rods (m).
        dio (FLOAT): Inner diameter of other rods (m).
        E (FLOAT): Elastic modulus of the axle material.
        nu (FLOAT): Poisson’s ratio of the axle material.
        mesh (MESH): Mesh containing node and cell information.
        uh (TENSOR): Post-processed displacement vector.
        ele_indices (TENSOR):  Number of bar elements,If None, uses all cells.

        Outputs:
            strain (TENSOR): Strain of the bar elements.
            stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "桁架塔应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点基于线弹性理论，对桁架塔结构的杆件执行应变–应力计算。
            节点通过单元网格、材料参数以及位移场，计算相应的单元应变及应力，用于结构后处理与安全性分析。
            并且用户可选择特定单元进行计算，或对所有单元执行统一的应变–应力分析。"""
            
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 1,  desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, 1,  desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, 1,  desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, 1,  desc="其他杆件的内径", title="其他杆内径"),
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量", title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比", title="泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="经过后处理的位移向量", title="位移向量"),
        PortConf("ele_num", DataType.INT, 0, desc="杆件单元个数，若为 None，则对全部单元进行计算。",
                 title="单元个数", default=None),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        
        model = TrussTowerData3D(
            dov=options.get("dov"),
            div=options.get("div"),
            doo=options.get("doo"),
            dio=options.get("dio")
        )
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )
        strain, stress = material.compute_strain_and_stress(
                        options.get("mesh"),
                        options.get("uh"),
                        ele_indices=options.get("ele_num"))

        return strain, stress
from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarMaterial"]


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
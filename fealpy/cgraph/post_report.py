from typing import Union
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["SolidReport"]


class SolidReport(CNodeType):
    r"""Solid Mechanics Computation Report Node.
    
    Inputs:
        mesh (MESH): The computational mesh object used in the simulation.
        material (OBJECT): Material properties, including Young’s modulus (E),
            Poisson’s ratio (nu), shear modulus (G), and Lamé parameters.
        uh (TENSOR): Displacement results in tensor form.
        uh_graph (OBJECT): Displacement visualization graph or figure.
        strain (TENSOR): Computed strain field tensor.
        strain_graph (OBJECT): Strain visualization graph or figure.
        stress (TENSOR): Computed stress field tensor.
        stress_graph (OBJECT): Stress visualization graph or figure.
        
    Outputs:
        report (pdf): Generated PDF report file.
    
    """
    TITLE: str = "固体力学计算报告"
    PATH: str = "后处理.报告生成"
    DESC: str = ""
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="计算网格对象", title="网格"),
        PortConf("material", DataType.NONE, 2, desc="场景所有的材料属性，包含弹性模量、泊松比、剪切模量、拉梅系数等", title="材料属性"),
        PortConf("uh", DataType.TENSOR, 1, desc="位移结果", title="位移数值解"),
        PortConf("uh_graph", DataType.NONE, 1, desc="位移云图", title="位移云图"),
        PortConf("strain", DataType.TENSOR, 1, desc="应变结果", title="应变数值解"),
        PortConf("strain_graph", DataType.NONE, 1, desc="应变云图", title="应变云图"),
        PortConf("stress", DataType.TENSOR, 1, desc="应力结果", title="应力数值解"),
        PortConf("stress_graph", DataType.NONE, 1, desc="应力云图", title="应力云图")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("report", DataType.NONE, title="计算报告")
    ]

    @staticmethod
    def run(mesh, material, uh, uh_graph, strain, strain_graph, stress, stress_graph):
        

        return report
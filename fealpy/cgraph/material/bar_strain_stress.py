from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25StrainStress",
           "Bar942StrainStress",
           "TrussTowerStrainStress"]

class Bar25StrainStress(CNodeType):
    r"""compute Strain and Stress for Bar Elements.
    
    Inputs:
        E (FLOAT): Elastic modulus of the axle material.
        nu (FLOAT): Poisson’s ratio of the axle material.
        mesh (MESH): Mesh containing node and cell information.
        uh (TENSOR): Post-processed displacement vector.
        coord_transform (TENSOR): Coordinate transformation matrix.

    Outputs:
        strain (TENSOR): Strain of the bar elements.
        stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "25杆应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点基于线弹性理论,对25杆桁架结构的杆件执行应变-应力计算。"""
            
    INPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量", title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比", title="泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移", title="全局位移"),
        PortConf("coord_transform", DataType.TENSOR, 1, desc="坐标变换矩阵", title="坐标变换")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.bar_data25_3d import BarData25
        from fealpy.csm.material import BarMaterial

        model = BarData25()
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )
        
        uh = options.get("uh").reshape(-1, 3)
        strain, stress = material.compute_strain_and_stress(
                        options.get("mesh"),
                        uh,
                        options.get("coord_transform"),
                        ele_indices=None)

        return strain, stress
    

class Bar942StrainStress(CNodeType):
    r"""compute Strain and Stress for Bar Elements.
    
    Inputs:
        E (FLOAT): Elastic modulus of the axle material.
        nu (FLOAT): Poisson’s ratio of the axle material.
        mesh (MESH): Mesh containing node and cell information.
        uh (TENSOR): Post-processed displacement vector.
        coord_transform (TENSOR): Coordinate transformation matrix.

    Outputs:
        strain (TENSOR): Strain of the bar elements.
        stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "942杆应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点基于线弹性理论，对942杆桁架结构的杆件执行应变–应力计算。"""
            
    INPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量", title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比", title="泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移", title="全局位移"),
        PortConf("coord_transform", DataType.TENSOR, 1, desc="坐标变换矩阵", title="坐标变换")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.bar_data942_3d import BarData942
        from fealpy.csm.material import BarMaterial

        model = BarData942()
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )
        
        uh = options.get("uh").reshape(-1, 3)
        strain, stress = material.compute_strain_and_stress(
                        options.get("mesh"),
                        uh,
                        options.get("coord_transform"),
                        ele_indices=None)

        return strain, stress


class TrussTowerStrainStress(CNodeType):
    r"""compute Strain and Stress for Bar Elements.
    
    Inputs:
        E (FLOAT): Elastic modulus of the axle material.
        nu (FLOAT): Poisson’s ratio of the axle material.
        mesh (MESH): Mesh containing node and cell information.
        uh (TENSOR): Post-processed displacement vector.
        coord_transform (TENSOR): Coordinate transformation matrix.

    Outputs:
        strain (TENSOR): Strain of the bar elements.
        stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "桁架塔应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点基于线弹性理论，对桁架塔结构的杆件执行应变–应力计算。"""
            
    INPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量", title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比", title="泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移", title="全局位移"),
        PortConf("coord_transform", DataType.TENSOR, 1, desc="坐标变换矩阵", title="坐标变换")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        
        model = TrussTowerData3D()
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )
        
        uh = options.get("uh").reshape(-1, 3)
        strain, stress = material.compute_strain_and_stress(
                        options.get("mesh"),
                        uh,
                        options.get("coord_transform"),
                        ele_indices=None)

        return strain, stress
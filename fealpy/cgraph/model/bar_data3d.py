from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25Data", 
           "Bar942Data",
           "TrussTower3d"]


class Bar25Data(CNodeType):
    r"""A 25-bar truss structure model.

    This node generates the node coordinates and cell connectivity for a 
    classic 25-bar space truss.

    Outputs:
        GD (int): Geometric dimension of the model (3D).
        A (float): Cross-sectional area of all bars (mm²).
        external_load (tensor): Global load vector applied at top nodes.
        dirichlet_dof (tensor): Boolean array indicating Dirichlet boundary DOFs.
        dirichlet_bc (tensor): Prescribed displacement values for all DOFs.
        
    """
    TITLE: str = "25杆模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点用于建立经典25杆空间桁架结构的几何模型。"""

    INPUT_SLOTS = []

    OUTPUT_SLOTS = [
        PortConf("GD", DataType.INT, desc="模型的几何维数", title="几何维数"),
        PortConf("A", DataType.FLOAT, desc="所有杆件的横截面积", title="横截面积"),
        PortConf("external_load", DataType.TENSOR, desc="施加在顶部节点的全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.TENSOR, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
        PortConf("dirichlet_bc", DataType.TENSOR, desc="边界节点的位移约束值", title="边界位移的值")
    ]

    @staticmethod
    def run():
        from fealpy.csm.model.truss.bar_data25_3d import BarData25
        
        model = BarData25()
        GD = model.GD
        A = model.A
        
        external_load = model.load()
        dirichlet_dof = model.is_dirichlet_boundary()
        dirichlet_bc = model.dirichlet_bc()

        return (GD, A, external_load,
                dirichlet_dof, dirichlet_bc)


class Bar942Data(CNodeType):
    r"""A 942-bar truss structure model.

    This node generates the node coordinates and cell connectivity for a 
    classic 942-bar space truss.

    Outputs:
        GD (int): Geometric dimension of the model (3D).
        A (float): Cross-sectional area of all bars (mm²).
        external_load (tensor): Global load vector applied at top nodes.
        dirichlet_dof (tensor): Boolean array indicating Dirichlet boundary DOFs.
        dirichlet_bc (tensor): Prescribed displacement values for all DOFs.

    """
    TITLE: str = "942杆模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点用于建立942杆空间桁架结构的几何模型。"""

    INPUT_SLOTS = []

    OUTPUT_SLOTS = [
        PortConf("GD", DataType.INT, desc="模型的几何维数", title="几何维数"),
        PortConf("A", DataType.FLOAT, desc="所有杆件的横截面积", title="横截面积"),
        PortConf("external_load", DataType.TENSOR, desc="施加在顶部节点的全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.TENSOR, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
        PortConf("dirichlet_bc", DataType.TENSOR, desc="边界节点的位移约束值", title="边界位移的值")
    ]

    @staticmethod
    def run():
        from fealpy.csm.model.truss.bar_data942_3d import BarData942

        model = BarData942()
        
        external_load = model.load()
        dirichlet_dof = model.is_dirichlet_boundary()
        dirichlet_bc = model.dirichlet_bc()

        return (model.GD, model.A, external_load,
                dirichlet_dof, dirichlet_bc)

class TrussTower3d(CNodeType):
    r"""3D Truss Tower Model.
    
     Inputs:
        dov (float): Outer diameter of vertical rods (m).
        div (float): Inner diameter of vertical rods (m).
        doo (float): Outer diameter of other rods (m).
        dio (float): Inner diameter of other rods (m).
        load (float): Total vertical load applied at top nodes (N).
        
    Outputs:
        GD (INT): Geometric dimension of the model.
        dov (float): Outer diameter of vertical rods (m).
        div (float): Inner diameter of vertical rods (m).
        doo (float): Outer diameter of other rods (m).
        dio (float): Inner diameter of other rods (m).
        external_load (Function): Global load vector.
        dirichlet_dof (Function): Dirichlet boundary DOF indices.
    """
    TITLE: str = "桁架塔模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点用于建立三维桁架塔的基础物理模型。通过输入杆件的外径、内径以及整体载荷等参数，
            计算竖向与其他类型杆件的截面积、惯性矩，以及结构在不同方向上的整体惯性属性。同时，
            它根据塔顶总载荷生成全局载荷向量，并构造底部固定约束的边界自由度函数，
            为后续有限元离散与求解提供所有必要的材料与几何信息。"""
            
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 0,  desc="竖向杆件的外径", title="竖杆外径", default=0.015),
        PortConf("div", DataType.FLOAT, 0,  desc="竖向杆件的内径", title="竖杆内径", default=0.010),
        PortConf("doo", DataType.FLOAT, 0,  desc="其他杆件的外径", title="其他杆外径", default=0.010),
        PortConf("dio", DataType.FLOAT, 0,  desc="其他杆件的内径", title="其他杆内径", default=0.007),
        PortConf("load", DataType.FLOAT, 1,  desc="施加在塔顶节点的总竖向载荷", title="总载荷", default=84820.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("GD", DataType.INT,  desc="模型的几何维数", title="几何维数"),
        PortConf("dov", DataType.FLOAT, desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, desc="其他杆件的内径", title="其他杆内径"),
        PortConf("external_load", DataType.TENSOR, desc="全局载荷向量，表示总载荷如何分布到顶部节点", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, desc="Dirichlet边界条件的自由度索引", title="边界自由度")
    ]


    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        
        model = TrussTowerData3D(
            dov=options.get("dov"),
            div=options.get("div"),
            doo=options.get("doo"),
            dio=options.get("dio")
        )
        
        load = options.get("load")
        external_load = model.external_load(load_total=load)

        return (model.GD, model.dov, model.div, model.doo, model.dio, 
           external_load, model.dirichlet_dof())
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarData", 
           "TrussTower3d"]


class BarData(CNodeType):
    r"""Bar truss structure model (25-bar or 942-bar).

    This node generates the node coordinates and cell connectivity for 
    classic bar space truss structures.
    
    Inputs:
        bar_type(MENU): Type of bar structure to generate.

    Outputs:
        GD (int): Geometric dimension of the model (3D).
        external_load (tensor): Global load vector applied at top nodes.
        dirichlet_dof (tensor): Boolean array indicating Dirichlet boundary DOFs.
        dirichlet_bc (tensor): Prescribed displacement values for all DOFs.
        
    """
    TITLE: str = "杆件桁架模型"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("bar_type", DataType.MENU, 0, desc="桁架结构类型", title="结构类型", 
                 default="bar25", items=["bar25", "bar942"])
    ]
    OUTPUT_SLOTS = [
        PortConf("GD", DataType.INT, desc="模型的几何维数", title="几何维数"),
        PortConf("external_load", DataType.TENSOR, desc="施加在顶部节点的全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.TENSOR, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
        PortConf("dirichlet_bc", DataType.TENSOR, desc="边界节点的位移约束值", title="边界位移的值")
    ]

    @staticmethod
    def run(**options):
        bar_type = options.get("bar_type")
        if bar_type == "bar25":
            from fealpy.csm.model.truss.bar_data25_3d import BarData25
            model = BarData25()
        elif bar_type == "bar942":
            from fealpy.csm.model.truss.bar_data942_3d import BarData942
            model = BarData942()
        else:
            raise ValueError(f"Unsupported bar_type: {bar_type}")
        
        external_load = model.load()
        dirichlet_dof = model.is_dirichlet_boundary()
        dirichlet_bc = model.dirichlet_bc()

        return (model.GD, external_load,
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
        external_load (tensor): Global load vector.
        dirichlet_dof (tensor): Dirichlet boundary DOF indices.
        dirichlet_bc (tensor): Prescribed displacement values for boundary DOFs.
    """
    TITLE: str = "桁架塔模型"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 0,  desc="竖向杆件的外径", title="竖杆外径", default=0.015),
        PortConf("div", DataType.FLOAT, 0,  desc="竖向杆件的内径", title="竖杆内径", default=0.010),
        PortConf("doo", DataType.FLOAT, 0,  desc="其他杆件的外径", title="其他杆外径", default=0.010),
        PortConf("dio", DataType.FLOAT, 0,  desc="其他杆件的内径", title="其他杆内径", default=0.007),
        PortConf("load", DataType.FLOAT, 0,  desc="施加在塔顶节点的总竖向载荷", title="总载荷", default=84820.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("GD", DataType.INT,  desc="模型的几何维数", title="几何维数"),
        PortConf("dov", DataType.FLOAT, desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, desc="其他杆件的内径", title="其他杆内径"),
        PortConf("external_load", DataType.TENSOR, desc="全局载荷向量，表示总载荷如何分布到顶部节点", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.TENSOR, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
         PortConf("dirichlet_bc", DataType.TENSOR, desc="边界位移约束值", title="边界位移值")
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
        dirichlet_dof = model.is_dirichlet_boundary()
        dirichlet_bc = model.dirichlet_bc()

        return (model.GD, model.dov, model.div, model.doo, model.dio, 
           external_load, dirichlet_dof, dirichlet_bc)
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["Rbar2d", "Rbar3d", "Rbeam3d", "RbeamAxle3d"]


class Rbar2d(CNodeType):
    r"""Coordinate Transformation for 2D Bar Elements.

    This node computes the coordinate transformation matrix for 2D bar elements
    from local to global coordinate systems.

    Inputs:
        mesh (MESH): The mesh object containing node and element information.

    Outputs:
        R (TENSOR): The coordinate transformation matrix of shape (NC, 2, 4),
                    where NC is the number of elements.
    """
    TITLE: str = "2D杆件坐标变换"
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("R", DataType.TENSOR, desc="坐标变换矩阵", title="坐标变换矩阵"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.utils import CoordTransform
        mesh = options.get("mesh")
        
        coord_transform = CoordTransform(method="bar2d")
        R = coord_transform.coord_transform_bar2d(mesh)
        return R
    
    
class Rbar3d(CNodeType):
    r"""Coordinate Transformation for 3D Bar Elements.

    This node computes the coordinate transformation matrix for 3D bar elements
    from local to global coordinate systems.

    Inputs:
        mesh (MESH): The mesh object containing node and element information.

    Outputs:
        R (TENSOR): The coordinate transformation matrix of shape (NC, 2, 6),
                    where NC is the number of elements.
    """
    TITLE: str = "3D杆件坐标变换"
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("R", DataType.TENSOR, desc="坐标变换矩阵", title="坐标变换矩阵"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.csm.utils import CoordTransform
        mesh = options.get("mesh")
    
        coord_transform = CoordTransform(method="bar3d")
        R = coord_transform.coord_transform_bar3d(mesh)
        return R


class Rbeam3d(CNodeType):
    r"""Coordinate Transformation for 3D Beam Elements.

    This node computes the coordinate transformation matrix for 3D beam elements
    from local to global coordinate systems.

    Inputs:
        mesh (MESH): The mesh object containing node and element information.
        vref (MENU): Reference vector for defining local coordinate system.

    Outputs:
        R (TENSOR): The coordinate transformation matrix of shape (NC, 12, 12),
                    where NC is the number of elements.
    """
    TITLE: str = "3D梁单元坐标变换" 
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("vref", DataType.MENU, 0, desc="参考向量，用于定义局部坐标系", title="参考向量", default=[0, 1, 0],
                 items=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ]
    OUTPUT_SLOTS = [
        PortConf("R", DataType.TENSOR, desc="坐标变换矩阵", title="坐标变换矩阵"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.csm.utils import CoordTransform
        mesh = options.get("mesh")
        vref = options.get("vref")
        
        coord_transform = CoordTransform(method="beam3d")
        R = coord_transform.coord_transform_beam3d(mesh, vref)
        return R


class RbeamAxle3d(CNodeType):
    r"""Coordinate Transformation for 3D Beam-Axle Coupled Elements.
    
    Inputs:
        mesh (MESH): The mesh object containing node, element and celldata['type'].
                     celldata['type']: 0=beam elements, 1=axle/spring elements.
        vref (MENU): Reference vector for defining local coordinate system.

    Outputs:
        R_beam (TENSOR): Coordinate transformation matrix for beam elements (NC_beam, 12, 12).
        R_axle (TENSOR): Coordinate transformation matrix for axle elements (NC_axle, 12, 12).
    """
    TITLE: str = "梁-轴耦合单元坐标变换"
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("vref", DataType.MENU, 0, desc="参考向量，用于定义局部坐标系", title="参考向量", 
                 default=[0, 1, 0], items=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ]
    
    OUTPUT_SLOTS = [
        PortConf("R_beam", DataType.TENSOR, desc="梁单元坐标变换矩阵", title="梁坐标变换"),
        PortConf("R_axle", DataType.TENSOR, desc="轴单元坐标变换矩阵", title="轴坐标变换")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.csm.utils import CoordTransform
        
        mesh = options.get("mesh")
        vref = options.get("vref")
        
        cell_types = mesh.celldata['type']
        
        # 获取梁单元和轴单元的索引
        beam_indices = bm.where(cell_types == 0)[0]
        axle_indices = bm.where(cell_types == 1)[0]
        
        coord_transform_beam = CoordTransform(method="beam3d")
        R_beam = coord_transform_beam.coord_transform_beam3d(mesh, vref, index=beam_indices)
        
        coord_transform_axle = CoordTransform(method="beam3d")
        R_axle = coord_transform_axle.coord_transform_beam3d(mesh, vref, index=axle_indices)
        return R_beam, R_axle
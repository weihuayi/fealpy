from typing import Type

from .nodetype import CNodeType, PortConf, DataType

__all__ = [
    "Rbar2d",
    "Rbar3d",
    "Rbeam3d"
]


class Rbar2d(CNodeType):
    r"""Coordinate Transformation for 2D Bar Elements.

    This node computes the coordinate transformation matrix for 2D bar elements
    from local to global coordinate systems.

    Inputs:
        mesh (MESH): The mesh object containing node and element information.
        index (TENSOR): Indices of elements to compute. Defaults to all elements.

    Outputs:
        R (TENSOR): The coordinate transformation matrix of shape (NC, 2, 4),
                    where NC is the number of elements.
    """
    TITLE: str = "2D杆件坐标变换"
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("index", DataType.TENSOR, 1, desc="单元索引，默认为所有单元", title="单元索引", default=None)
    ]
    OUTPUT_SLOTS = [
        PortConf("R", DataType.TENSOR, desc="坐标变换矩阵", title="坐标变换矩阵"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.utils import CoordTransform
        mesh = options.get("mesh")
        index = options.get("index")
        indices = index if index is not None else slice(index)
        
        coord_transform = CoordTransform(method="bar2d")
        R = coord_transform.coord_transform_bar2d(mesh, indices)
        return R
    
    
class Rbar3d(CNodeType):
    r"""Coordinate Transformation for 3D Bar Elements.

    This node computes the coordinate transformation matrix for 3D bar elements
    from local to global coordinate systems.

    Inputs:
        mesh (MESH): The mesh object containing node and element information.
        index (TENSOR): Indices of elements to compute. Defaults to all elements.

    Outputs:
        R (TENSOR): The coordinate transformation matrix of shape (NC, 2, 6),
                    where NC is the number of elements.
    """
    TITLE: str = "3D杆件坐标变换"
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("index", DataType.INT, 1, desc="单元索引，默认为所有单元", title="单元索引", default=None)
    ]
    OUTPUT_SLOTS = [
        PortConf("R", DataType.TENSOR, desc="坐标变换矩阵", title="坐标变换矩阵"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.csm.utils import CoordTransform
        mesh = options.get("mesh")
        index = options.get("index")
        indices = index if index is not None else slice(index)
        
        coord_transform = CoordTransform(method="bar3d")
        R = coord_transform.coord_transform_bar3d(mesh, indices)
        return R


class Rbeam3d(CNodeType):
    r"""Coordinate Transformation for 3D Beam Elements.

    This node computes the coordinate transformation matrix for 3D beam elements
    from local to global coordinate systems.

    Inputs:
        mesh (MESH): The mesh object containing node and element information.
        vref (MENU): Reference vector for defining local coordinate system.
        index (TENSOR): Indices of elements to compute. Defaults to all elements.

    Outputs:
        R (TENSOR): The coordinate transformation matrix of shape (NC, 12, 12),
                    where NC is the number of elements.
    """
    TITLE: str = "3D梁单元坐标变换" 
    PATH: str = "utils.coordtransform"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("vref", DataType.MENU, 0, desc="参考向量，用于定义局部坐标系", title="参考向量", default=[0, 1, 0],
                 items=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        PortConf("index", DataType.TENSOR, 1, desc="单元索引，默认为所有单元", title="单元索引", default=None)
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
        index = options.get("index")
        indices = index if index is not None else slice(index)
        
        coord_transform = CoordTransform(method="beam3d")
        R = coord_transform.coord_transform_beam3d(mesh, vref, indices)
        return R

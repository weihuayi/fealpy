
from typing import Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike


class CoordTransform:
    """Coordinate transformation for structural elements.
    
    This class provides methods to compute the coordinate transformation matrix
    for different types of structural elements (bars and beams) from local to
    global coordinate systems.
    """
    def __init__(self, 
                 method: Literal["bar3d", "beam2d", "beam3d", None] = None) -> None:
        """Initialize the coordinate transformation object.
        
        Parameters:
            method (Literal): The type of coordinate transformation method.
                - 'bar2d': 2D bar element (default)
                - 'bar3d': 3D bar element
                - 'beam2d': 2D beam element
                - 'beam3d': 3D beam element
        """
        self.method = method
        
    def __call__(self, mesh, vref=None, index: Index=_S) -> TensorLike:
        """Call the appropriate coordinate transformation method.
        
        Parameters:
            mesh: The mesh object.
            vref: Reference vector for defining local coordinate system(only for 3D elements).
            index (Index): The indices of elements. Defaults to all elements.
            
        Returns:
            TensorLike: The coordinate transformation matrix.
        """
        if self.method == 'beam3d':
            return self.coord_transform_beam3d(mesh, vref, index)
        elif self.method == 'beam2d':
            return self.coord_transform_beam2d(mesh, vref, index)
        elif self.method == 'bar3d':
            return self.coord_transform_bar3d(mesh, vref, index)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def coord_transform_bar2d(self, mesh, vref=None, index: Index=_S) -> TensorLike:
        """Construct the coordinate transformation matrix for 2D bar elements.
        
        Returns:
            TensorLike: The coordinate transformation matrix of shape (NC, 2, 4),
                where NC is the number of elements.

        Notes:
            2D bar elements only consider axial degrees of freedom.
           The transformation matrix has the form:
                R = [[m, n, 0, 0],
                    [0, 0, m, n]]
            where m, n are direction cosines:
                m = (x2 - x1) / L
                n = (y2 - y1) / L
        """
        node = mesh.entity('node')
        cell = mesh.entity('cell')[index]
        bar_nodes = node[cell]
        
        x, y = bar_nodes[..., 0], bar_nodes[..., 1]
        bars_length = mesh.entity_measure('cell')[index]

        m = (x[..., 1] - x[..., 0]) / bars_length
        n = (y[..., 1] - y[..., 0]) / bars_length

        NC = cell.shape[0]
        zeros = bm.zeros((NC, 2))
        
        row1 = bm.stack([m, n], axis=-1)  # shape: (NC, 2)
        row1 = bm.concatenate([row1, zeros], axis=-1)  # shape: (NC, 4)
        
        row2 = bm.stack([m, n], axis=-1)  # shape: (NC, 2)
        row2 = bm.concatenate([zeros, row2], axis=-1)  # shape: (NC, 4)
        
        R = bm.stack([row1, row2], axis=1)  # shape: (NC, 2, 4)
        
        return R
    
    def coord_transform_bar3d(self, mesh, vref=None, index: Index=_S)-> TensorLike:
        """Construct the coordinate transformation matrix for 3D bar elements.
        
        Returns:
            TensorLike: The coordinate transformation matrix of shape (NC, 2, 6),
                where NC is the number of elements.

        Notes:
            3D bar elements only consider axial degrees of freedom.
            The transformation matrix has the form:
                R = [[l, m, n, 0, 0, 0],
                    [0, 0, 0, l, m, n]]
            where l, m, n are direction cosines:
                l = (x2 - x1) / L
                m = (y2 - y1) / L
                n = (z2 - z1) / L   
        """
        node = mesh.entity('node')
        cell = mesh.entity('cell')[index]
        bar_nodes = node[cell]  # shape: (NC, 2, 3)
        
        x, y, z = bar_nodes[..., 0], bar_nodes[..., 1], bar_nodes[..., 2]
        bars_length = mesh.entity_measure('cell')[index]

        l = (x[..., 1] - x[..., 0]) / bars_length
        m = (y[..., 1] - y[..., 0]) / bars_length
        n = (z[..., 1] - z[..., 0]) / bars_length
        
        NC = cell.shape[0]
        zeros = bm.zeros((NC, 3))
        row1 = bm.stack([l, m, n], axis=-1)  # shape: (NC, 3)
        row1 = bm.concatenate([row1, zeros], axis=-1)  # shape: (NC, 6)
        
        row2 = bm.stack([l, m, n], axis=-1)  # shape: (NC, 3)
        row2 = bm.concatenate([zeros, row2], axis=-1)  # shape: (NC, 6)
        
        R = bm.stack([row1, row2], axis=1)  # shape: (NC, 2, 6)

        return R
    
    def coord_transform_beam2d(self, mesh, vref=None, index: Index=_S)-> TensorLike:
        """Construct the coordinate transformation matrix for 2D beam elements.
        
        Returns:
            TensorLike: The coordinate transformation matrix of shape (NC, 6, 6),
                where NC is the number of elements.

        Notes:
            2D beam elements have 3 degrees of freedom per node:
            2 translational and 1 rotational.
        """
        pass
    
    def coord_transform_beam3d(self, mesh, vref=None, index: Index=_S) -> TensorLike:
        """Construct the coordinate transformation matrix for 3D elements.
        
        Returns:
            TensorLike: The coordinate transformation matrix of shape (NC, 12, 12),
                where NC is the number of elements.

        Notes:
            3D beam elements have 6 degrees of freedom per node: 3 translational and 3 rotational.
        """
        if vref is None:
            vref = [0, 1, 0]  # 默认参考向量
            
        node= mesh.entity('node')
        cell = mesh.entity('cell')[index]
        bar_nodes = node[cell]
        
        x, y, z = bar_nodes[..., 0], bar_nodes[..., 1], bar_nodes[..., 2]
        bars_length = mesh.entity_measure('cell')[index]

        # 第一行（轴向单位向量）
        T11 = (x[..., 1] - x[..., 0]) / bars_length
        T12 = (y[..., 1] - y[..., 0]) / bars_length
        T13 = (z[..., 1] - z[..., 0]) / bars_length
        
        k1, k2, k3 = vref

        # 第二行（局部y方向）
        A = bm.sqrt((T12 * k3 - T13 * k2)**2 + 
                    (T13 * k1 - T11 * k3)**2 +
                    (T11 * k2 - T12 * k1)**2)
       
        T21 = -(T12 * k3 - T13 * k2) / A
        T22 = -(T13 * k1 - T11 * k3) / A
        T23 = -(T11 * k2 - T12 * k1) / A

         # 第三行（局部z方向 = 第一行 × 第二行）
        B = bm.sqrt((T12 * T23 - T13 * T22)**2 +
                    (T13 * T21 - T11 * T23)**2 +
                    (T11 * T22 - T12 * T21)**2)
        
        T31 = (T12 * T23 - T13 * T22) / B
        T32 = (T13 * T21 - T11 * T23) / B
        T33 = (T11 * T22 - T12 * T21) / B
        
        # 构造3x3基础旋转矩阵 T0
        T0 = bm.stack([
                    bm.stack([T11, T12, T13], axis=-1),  # shape: (NC, 3)
                    bm.stack([T21, T22, T23], axis=-1),
                    bm.stack([T31, T32, T33], axis=-1)
                ], axis=1)  # shape: (NC, 3, 3)
        
        # 构造12x12旋转变换矩阵 R
        NC = T0.shape[0]
        O = bm.zeros((NC, 3, 3))
        row1 = bm.concatenate([T0   , O,  O,  O], axis=2)
        row2 = bm.concatenate([O,  T0, O,  O], axis=2)
        row3 = bm.concatenate([O,  O,  T0, O], axis=2)
        row4 = bm.concatenate([O,  O,  O,  T0], axis=2)

        R = bm.concatenate([row1, row2, row3, row4], axis=1)  #shape: (NC, 12, 12)
        return R
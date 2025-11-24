
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike


def coord_transform(mesh, vref=[0, 1, 0], index: Index=_S) -> TensorLike:
        """Construct the coordinate transformation matrix for 3D elements.
        
        Parameters:
            mesh (Mesh): The mesh object.
            vref (TensorLike): A reference vector to define the local y-axis direction.
            index (Index): The indices of the elements to compute the transformation matrix for.
                If None, compute for all elements. Defaults to _S (all elements).
    
        Returns:
            R(TensorLike): The coordinate transformation matrix.
        """
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
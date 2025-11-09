from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S

from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.fem.integrator import LinearInt, OpInt, CellInt, enable_cache


class BarIntegrator(LinearInt, OpInt, CellInt):
    """
    Integrator for 3D bar (truss) element stiffness.

    Assumes TensorFunctionSpace uses a component-blocked layout with shape (-1, GD).

    Parameters:
        space (_FS): The function space.
        model: PDE model.
        material: Material properties.
        index (Index, optional): Index for integration.
        method (str, optional): Integration method.

    Methods:
        to_global_dof(space): Returns the mapping from cell to global DOF.
        assembly(space): Assembles and returns the local stiffness matrices.
    """
    def __init__(self, 
                 space: _FS, 
                 model,
                 material, 
                 index: Index=_S,
                 method: Optional[str]=None )-> None:
        super().__init__()

        self.space = space
        self.model = model
        self.material = material
        self.index = index
        
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        """Returns the mapping from cell to global DOF for selected cells."""
        cell2dof = space.cell_to_dof()  # (NC, ldof)
        index = self.index
         # 如果 index 是 slice 对象 (例如 _S 即 slice(None))
        if isinstance(index, slice):
            mesh = space.mesh
            NC = mesh.number_of_cells()
            index = bm.arange(NC)[index]  # 转换为数组
        return cell2dof[index]  # (NC_selected, ldof)
        
        
    def assembly(self, space: _FS) -> TensorLike:
        """Assembles the local stiffness matrices for all bar elements.

        Parameters:
            space (_FS): The function space.

        Returns:
            TensorLike: Local stiffness matrices for each element.
        """
        mesh = space.mesh
        GD = 3
        E = self.material.E
        
        index = self.index
        NC = mesh.number_of_cells()
        
        # 如果 index 是 slice 对象 (例如 _S 即 slice(None))
        if isinstance(index, slice):
            index = bm.arange(NC)[index]  # 转换为数组

        A = self.model.A  # (NC,) cross-sectional areas for selected elements
        if isinstance(A, (int, float)):
            # A 是标量,所有单元相同
            A_selected = A  
        else:
            # A 是数组,取出对应子集
            A_selected = A[index]  # (NC_selected,)
    
        # 只计算选定单元的长度和方向
        l = mesh.edge_length()[index].reshape(-1, 1)  # (NC_selected, 1)
        tan = mesh.edge_tangent()[index]  # (NC_selected, 3)
        unit_tan = tan / l  # (NC_selected, 3)

        R = bm.einsum('ik,im->ikm', unit_tan, unit_tan)  # (NC, 3, 3)

        NC = index.shape[0]  # Number of selected cells
        k = bm.zeros((NC, GD*2, GD*2), dtype=bm.float64)
        k[:, :GD, :GD] = R
        k[:, -GD:, :GD] = -R
        k[:, :GD, -GD:] = -R
        k[:, -GD:, -GD:] = R
        
        EA = E * A_selected  # 标量或 (NC_selected,)
        if not isinstance(EA, (int, float)):
            # EA 是数组,需要 reshape
            k *= EA[:, None, None]  # 广播到 (NC_selected, 6, 6)
        else:
            # EA 是标量,直接相乘
            k *= EA
            
        k /= l[:, None]  # l 已经是 (NC_selected, 1), 可以直接广播
        return k
from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S
from fealpy.decorator.variantmethod import variantmethod

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
                method: Literal['geometric', None] = None)-> None:
        super().__init__()

        self.space = space
        self.model = model
        self.material = material
        self.index = index
        
        self.assembly.set(method)
        
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
        
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Assembles the local stiffness matrices for all bar elements.

        Parameters:
            space (_FS): The function space.

        Returns:
            TensorLike: Local stiffness matrices for each element.
        """
        
        GD = self.model.GD
        E = self.material.E
        
        index = self.index
        mesh = space.mesh
        NC = mesh.number_of_cells()
        
        if isinstance(index, slice):
            index = bm.arange(NC)[index]  # 转换为数组

        A = self.model.A  # (NC,) cross-sectional areas for selected elements
        if isinstance(A, (int, float)):
            A_selected = A  
        else:
            A_selected = A[index]  # (NC_selected,)
    
        l = mesh.edge_length()[index].reshape(-1, 1)  # (NC_selected, 1)
        tan = mesh.edge_tangent()[index]  # (NC_selected, 3)
        unit_tan = tan / l  # (NC_selected, 3)

        R = bm.einsum('ik, im -> ikm', unit_tan, unit_tan)  # (NC, 3, 3)

        NC_selected = index.shape[0]  # Number of selected cells
        k = bm.zeros((NC_selected, GD*2, GD*2), dtype=bm.float64)
        k[:, :GD, :GD] = R
        k[:, -GD:, :GD] = -R
        k[:, :GD, -GD:] = -R
        k[:, -GD:, -GD:] = R
        
        EA = E * A_selected  # 标量或 (NC_selected,)
        if not isinstance(EA, (int, float)):
            k *= EA[:, None, None]  # (NC_selected, 6, 6)
        else:
            k *= EA
            
        k /= l[:, None] 
         
        return k
    
    @assembly.register('geometric')
    def assembly(self, space: _FS, sigma: TensorLike) -> TensorLike:
        """Assembles the geometric stiffness matrices for all bar elements (for buckling analysis).

        Parameters:
            space (_FS): The function space.
            sigma (TensorLike): The stress tensor.

        Returns:
            TensorLike: Geometric stiffness matrices for each element, shape (NC_selected, 6, 6).
        """
        
        GD = self.model.GD
        
        index = self.index
        mesh = space.mesh
        NC = mesh.number_of_cells()
        
        if isinstance(index, slice):
            index = bm.arange(NC)[index] 
            
        if isinstance(sigma, (int, float)):
            sigma_selected = sigma
        else:
            sigma_selected = sigma[index]
            
        A = self.model.A
        if isinstance(A, (int, float)):
            A_selected = A
        else:
            A_selected = A[index]  # (NC_selected,)
            
        #  Axial force in each element: N = sigma * A
        N = bm.einsum('ci, c -> c', sigma_selected, A_selected)  # (NC_selected,)
        l = mesh.edge_length()[index].reshape(-1, 1) 
        tan = mesh.edge_tangent()[index]  # (NC_selected, GD)
        unit_tan = tan / l  

        R = bm.einsum('ik, im -> ikm', unit_tan, unit_tan)  # (NC, GD, GD)
        I = bm.eye(GD, dtype=bm.float64)
        P = I[None, :, :] - R 
        
        NC_selected = index.shape[0]
        k = bm.zeros((NC_selected, GD*2, GD*2), dtype=bm.float64)
        
        
        k[:, :GD, :GD] = P
        k[:, :GD, -GD:] = -P
        k[:, -GD:, :GD] = -P
        k[:, -GD:, -GD:] = P
        
        if not isinstance(N, (int, float)):
            k *= (N[:, None, None] / l[:, None])  # (NC_selected, 1, 1)
        else:
            k *= N / l
            
        return k
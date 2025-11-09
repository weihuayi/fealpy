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
        material: Material properties with attributes E and A.
        index (Index, optional): Index for integration.
        method (str, optional): Integration method.

    Attributes:
        space (_FS): The function space.
        material: Material properties.
        index (Index): Integration index.

    Methods:
        to_global_dof(space): Returns the mapping from cell to global DOF.
        assembly(space): Assembles and returns the local stiffness matrices.
    """
    def __init__(self, 
                 space: _FS, 
                 material, 
                 index: Index=_S,
                 method: Optional[str]=None )-> None:
        super().__init__()

        self.space = space
        self.material = material
        self.index = index
        
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        """
        Returns the mapping from cell to global DOF.

        Parameters:
            space (_FS): The function space.

        Returns:
            TensorLike: The cell-to-DOF mapping.
        """
        return space.cell_to_dof()

    def assembly(self, space: _FS) -> TensorLike:
        """
        Assembles the local stiffness matrices for all bar elements.

        Parameters:
            space (_FS): The function space.

        Returns:
            TensorLike: Local stiffness matrices for each element.
        """
        mesh = space.mesh
        GD = 3
        self.E = self.material.E
        self.A = self.material.A

        l = mesh.edge_length().reshape(-1, 1)
        tan = mesh.edge_tangent()
        unit_tan = tan / l

        R = bm.einsum('ik,im->ikm', unit_tan, unit_tan)

        NE = mesh.number_of_edges()
        k = bm.zeros((NE, GD*2, GD*2), dtype=bm.float64)
        k[:, :GD, :GD] = R
        k[:, -GD:, :GD] = -R
        k[:, :GD, -GD:] = -R
        k[:, -GD:, -GD:] = R
        k *= (self.E * self.A)
        k /= l[:, None]

        return k
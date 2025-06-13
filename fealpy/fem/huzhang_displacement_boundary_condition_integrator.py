
from typing import Optional, Literal

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, SourceLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral
from .integrator import LinearInt, enable_cache

class HuZhangDisplacementBoundaryConditionIntegrator(LinearInt): 
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, g: Optional[SourceLike]=None, q: int=None) -> None:
        super().__init__()
        self.g = g
        self.q = q 

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        bdface = mesh.boundary_face_index()
        f2c = mesh.face_to_cell()[bdface]
        cell2dof = space.cell_to_dof()[f2c[:, 0]]
        return cell2dof 

    @enable_cache
    def fetch(self, space: _FS, /, inidces=None):
        pass

    def assembly(self, space: _FS, indices=None) -> TensorLike:
        '''
        Notes
        -----
        Only support 3D case.
        '''
        return b 



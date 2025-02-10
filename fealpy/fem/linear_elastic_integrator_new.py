from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh, SimplexMesh, TensorMesh
from ..functionspace.space import FunctionSpace as _FS
from ..functionspace.tensor_space import TensorFunctionSpace as _TS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod
)
from fealpy.fem.utils import SymbolicIntegration

class LinearElasticIntegrator(LinearInt, OpInt, CellInt):
    """
    The linear elastic integrator for function spaces based on homogeneous meshes.
    """
    def __init__(self, 
                 material,
                 q: Optional[int]=None, *,
                 index: Index=_S,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.material = material
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

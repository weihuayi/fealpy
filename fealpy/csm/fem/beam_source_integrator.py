from fealpy.backend import backend_manager as bm
from fealpy.fem.integrator import LinearInt, SrcInt, CellInt, enable_cache
from fealpy.typing import TensorLike, Index, _S, SourceLike
from typing import Optional, Literal
from fealpy.functionspace.space import FunctionSpace as _FS

class BeamSourceIntegrator(LinearInt, SrcInt, CellInt):
    """
    Integrator for computing source terms for various types of beam elements

    Supports:
    - Pure bending beam: 2 DOFs/node [w, theta]
    - 2D beam: 3 DOFs/node [u, w, theta]
    - 3D beam: 6 DOFs/node [u, v, w, theta_x, theta_y, theta_z]

    Parameters
    ----------
    space : FunctionSpace
        Finite element function space.
    beam_type : str
        Beam type: 'pure', '2d', or '3d'
    source : SourceLike
        Source term function
    l : array_like
        Beam element lengths (NC,)
    method : str, optional
        Integration method (default 'assembly')
    """
    def __init__(self, 
                 space: _FS,
                 beam_type: Literal['pure', '2d', '3d'],
                 source: Optional[SourceLike] = None,
                 l: TensorLike = None,
                 method: Optional[str] = None) -> None:
        self.space = space
        self.type = beam_type.lower()
        self.ft = source
        self.l = l
        self.NC = l.shape[0]
        method = 'assembly' if method is None else method
        super().__init__(method=method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()

    def assembly(self, space: _FS) -> TensorLike:
        if self.type == 'pure':
            return self._source_pure()
        elif self.type == '2d':
            return self._source_2d()
        elif self.type == '3d':
            return self._source_3d()
        else:
            raise ValueError(f"未知梁类型: {self.type}")

    def _source_pure(self) -> TensorLike:
        """
        Pure Bending Beam (2 DOFs per node: [w, theta])
        """
        l = self.l[:, None]  # shape (NC, 1)
        q = self.ft  
        f1 = q * l / 2
        f2 = q * l**2 / 12
        return bm.concatenate([f1, f2, f1, -f2], axis=-1)  # shape (NC, 4)

    def _source_2d(self) -> TensorLike:
        """
        2D Beam (3 DOFs per node: [u, w, theta])
        Uniform distributed load q = [fx, fy], where fx is the axial load and fy is the vertical load.
        """
        l = self.l[:, None]
        q = self.ft  
        fx = q[:, 0:1]
        fy = q[:, 1:2]

        fx1 = fx * l / 2
        fy1 = fy * l / 2
        fy2 = fy * l**2 / 12

        return bm.concatenate([
            fx1, fy1, fy2,   # node 1: u, w, theta
            fx1, fy1, -fy2   # node 2: u, w, theta
        ], axis=-1)  # shape (NC, 6)


    def _source_3d(self) -> TensorLike:
        """
        3D Beam (6 DOFs per node: [u, v, w, theta_x, theta_y, theta_z])
        Uniform distributed load q = [fx, fy, fz], where fx, fy, fz are the distributed loads in the local x, y, z directions, respectively.
        """
        l = self.l[:, None]
        q = self.ft  
        fx = q[:, 0:1]
        fy = q[:, 1:2]
        fz = q[:, 2:3]

        fx1 = fx * l / 2

        fy1 = fy * l / 2
        fy2 = fy * l**2 / 12

        fz1 = fz * l / 2
        fz2 = fz * l**2 / 12

        zero = bm.zeros_like(fx1)

        return bm.concatenate([
            fx1, fy1, fz1, zero,  fz2, -fy2,   # node 1: u, v, w, θx, θy, θz
            fx1, fy1, fz1, zero, -fz2,  fy2    # node 2: u, v, w, θx, θy, θz
        ], axis=-1)  # shape (NC, 12)


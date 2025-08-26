
from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, SourceLike
from fealpy.decorator import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, SrcInt, CellInt, enable_cache



class EulerBernoulliBeamPLSourceIntegrator(LinearInt, SrcInt, CellInt):
    """
    Beam element source term integrator.

    Parameters
        space : FunctionSpace
            Finite element space.
        beam_type : str
            Beam type: 'euler_bernoulli_2d', 'normal_2d', or 'euler_bernoulli_3d'.
        source : SourceLike
            Source term function.
        l : array_like
            Beam element length (NC,).
        method : str, optional
            Integration method (default 'assembly').
    """
    def __init__(self, 
                 space: _FS,
                 beam_type: Literal['euler_bernoulli_2d', 'normal_2d', 'euler_bernoulli_3d'],
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
        # Return mapping from element to global DOFs
        return space.cell_to_dof()

    @variantmethod("euler_bernoulli_2d")
    def assembly_pointload(self, space: _FS) -> TensorLike:
        """
        Assemble nodal concentrated force for 2D beam (2 DOFs per node: [u, w]).
        """
        q = self.ft  # shape (NC, 1)
        zero = bm.zeros_like(q)
        return bm.concatenate([zero, zero, q, zero], axis=-1)  # (NC, 4)

    @assembly_pointload.register("normal_2d")
    def assembly_pointload(self, space: _FS) -> TensorLike:
        """
        Assemble nodal concentrated force for 2D beam (3 DOFs per node: [u, w, theta]).
        """
        q = self.ft
        fx = q[:, 0:1]
        fy = q[:, 1:2]
        zero = bm.zeros_like(fx)
        return bm.concatenate([zero, zero, zero, fx, fy, zero], axis=-1)  # (NC, 6)

    @assembly_pointload.register("euler_bernoulli_3d")
    def assembly_pointload(self, space: _FS) -> TensorLike:
        """
        Assemble nodal concentrated force for 3D beam (6 DOFs per node: [u, v, w, θx, θy, θz]).
        """
        q = self.ft
        fx = q[:, 0:1]
        fy = q[:, 1:2]
        fz = q[:, 2:3]
        zero = bm.zeros_like(fx)
        return bm.concatenate([
            zero, zero, zero, zero, zero, zero,  # node 1
            fx, fy, fz, zero, zero, zero         # node 2
        ], axis=-1)  # (NC, 12)

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
        """
        Assemble the source term vector.
        Depending on the beam type ('pure', '2d', or '3d'), calls the corresponding source assembly method and returns the assembled source vector.
        Parameters
            space : _FS
            Finite element space object used to define the assembly environment.
        Returns
            source_vector : TensorLike
            Assembled source vector of type TensorLike, representing the source term for the corresponding beam type.
        Raises
            ValueError
            Raised if self.type is not 'pure', '2d', or '3d', indicating an unknown beam type.
        Notes
            This method automatically selects the appropriate source assembly implementation based on the beam type, making it easy to extend and maintain.
        """
    
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
        Pure bending beam (2 DOFs per node: [w, theta])
        Uniform distributed load q = [f], where f is the bending load.
        The source term is computed as:
        f1 = q * l / 2
        f2 = q * l**2 / 12
        Returns:
            TensorLike: Assembled source vector of shape (NC, 4) for pure bending beam.
        """
        l = self.l[:, None]  # shape (NC, 1)
        q = self.ft  
        f1 = q * l / 2
        f2 = q * l**2 / 12
        return bm.concatenate([f1, f2, f1, -f2], axis=-1)  # shape (NC, 4)

    def _source_2d(self) -> TensorLike:
        """
        2D Beam (3 DOFs per node: [u, w, theta])
        Uniform distributed load q = [fx, fy], where fx is the load in the x direction and fy is the load in the y direction.
        The source term is computed as:
        fx1 = fx * l / 2
        fy1 = fy * l / 2
        fy2 = fy * l**2 / 12
        Returns:
            TensorLike: Assembled source vector of shape (NC, 6) for 2D beam.
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
        3D Beam (6 DOFs per node: [u, v, w, θx, θy, θz])
        Uniform distributed load q = [fx, fy, fz], where fx, fy, fz are the loads in the x, y, and z directions respectively.
        The source term is computed as:
        fx1 = fx * l / 2
        fy1 = fy * l / 2
        fy2 = fy * l**2 / 12
        fz1 = fz * l / 2
        fz2 = fz * l**2 / 12
        Returns:
            TensorLike: Assembled source vector of shape (NC, 12) for 3D beam.
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



from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class IntracranialAneurysm3d():
    def __init__(self, options: dict = None):
        self.options = options
        self.mu = 0.004
        self.rho = 1.06e3

        from fealpy.mesher.intracranial_aneurysm_mesher import IntracranialAneurysm3dMesher
        mesher = IntracranialAneurysm3dMesher()
        self.mesh = mesher.generate_mesh()

        if options is not None:
            self.h = options.get('lc', 0.06)

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 3

    @cartesian
    def velocity_dirichlet(self, p:TensorLike) -> TensorLike:
        inlet = self.inlet_velocity(p)
        wall = self.wall_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_wall] = wall[is_wall]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)

    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        mesh = self.mesh
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.14 * 0.05136
        result[..., 1] = 0.14 * (-0.030365)
        result[..., 2] = 0.14 * 0.998218
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape[0], dtype=bm.float64)
        return result
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on wall."""
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    @cartesian
    def is_velocity_boundary(self, p):
        is_out = self.is_outlet_boundary(p)
        is_in = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        # return is_in | is_wall
        return ~is_out
        
    
    @cartesian
    def is_pressure_boundary(self, p : TensorLike = None) -> TensorLike:
        if p is None:
            return 1
        is_out = self.is_outlet_boundary(p)
        return is_out

    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        inlet = bm.unique(self.mesh.boundary_faces["cap_1"].reshape(-1))
        result = bm.zeros(p.shape[0], dtype=bm.bool)
        result[inlet] = True
        return result

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        outlet1 = bm.unique(self.mesh.boundary_faces["cap_2"].reshape(-1))
        outlet2 = bm.unique(self.mesh.boundary_faces["cap_3"].reshape(-1))
        outlet3 = bm.unique(self.mesh.boundary_faces["cap_4"].reshape(-1))
        outlet4 = bm.unique(self.mesh.boundary_faces["cap_5"].reshape(-1))
        outlet5 = bm.unique(self.mesh.boundary_faces["cap_6"].reshape(-1))
        result = bm.zeros(p.shape[0], dtype=bm.bool)
        result[outlet1] = True
        result[outlet2] = True
        result[outlet3] = True
        result[outlet4] = True
        result[outlet5] = True
        return result
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        inlet = bm.unique(self.mesh.boundary_faces["wall"].reshape(-1))
        result = bm.zeros(p.shape[0], dtype=bm.bool)
        result[inlet] = True
        return result
        
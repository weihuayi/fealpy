
from ..nodetype import CNodeType, PortConf, DataType


class DLDMicrofluidicChip3D(CNodeType):
    r"""Construct a 3D fluid mathematical model for a DLD (Deterministic Lateral Displacement) microfluidic chip.

    Inputs:
        thickness (float): Chip thickness in the z-direction.
        radius (float): Radius of each micropillar obstacle.
        centers (Tensor): Coordinates of micropillar centers in the channel plane.
        inlet_boundary (Tensor): Nodal or edge coordinates defining the inlet boundary.
        outlet_boundary (Tensor): Nodal or edge coordinates defining the outlet boundary.
        wall_boundary (Tensor): Nodal or edge coordinates defining the lateral wall boundaries.

    Outputs:
        velocity_dirichlet (function): Function defining Dirichlet boundary conditions for velocity.
        pressure_dirichlet (function): Function defining Dirichlet boundary conditions for pressure.
        is_velocity_boundary (function): Logical function that identifies velocity (no-slip or inlet) boundaries.
        is_pressure_boundary (function): Logical function that identifies pressure (outlet) boundaries.
    """
    TITLE: str = "三维 DLD 微流控芯片流体数学模型"
    PATH: str = "模型.DLD 微流控芯片"
    DESC: str = """该节点基于DLD微流控芯片几何参数构建三维流体数学模型, 定义速度与压力的Dirichlet边界条
                件及其识别函数，用于后续流场有限元或有限体积求解。"""
    INPUT_SLOTS = [
        PortConf("thickness", DataType.FLOAT, title = "芯片厚度"),
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱圆心坐标"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界"),
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.backend import TensorLike
        from fealpy.decorator import cartesian

        class DLD3D():
            def __init__(self, **options):
                self.eps = 1e-10
                self.mu = 1e-3
                self.rho = 1.0
                self.thickness = options.get('thickness', 0.1)
                self.radius = options.get('radius')
                self.centers = options.get('centers')
                self.inlet_boundary = options.get('inlet_boundary')
                self.outlet_boundary = options.get('outlet_boundary')
                self.wall_boundary = options.get('wall_boundary')

            def get_dimension(self) -> int: 
                """Return the geometric dimension of the domain."""
                return 3

            @cartesian
            def source(self, p: TensorLike) -> TensorLike:
                """Compute exact source """
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                result = bm.zeros(p.shape, dtype=bm.float64)
                return result

            @cartesian
            def inlet_velocity(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = 16*0.45*y*z*(1-y)*(0.1-z)/(0.41**4)
                return result
            
            @cartesian
            def outlet_pressure(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                result = bm.zeros(p.shape[0], dtype=bm.float64)
                return result
            
            @cartesian
            def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                bd = self.inlet_boundary
                return self.is_lateral_boundary(p, bd)  

            @cartesian
            def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where pressure is defined is on boundary."""
                bd = self.outlet_boundary
                return self.is_lateral_boundary(p, bd)
            
            @cartesian
            def is_wall_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                bd = self.wall_boundary
                return self.is_lateral_boundary(p, bd)
            
            @cartesian
            def is_top_or_bottom(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on top or bottom boundary."""
                atol = 1e-12
                thickness = self.thickness
                cond = (bm.abs(p[:, -1]) < atol) | (bm.abs(p[:, -1] - thickness) < atol)
                return cond
            
            @cartesian
            def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                x = p[..., 0]
                y = p[..., 1]
                radius = self.radius
                atol = 1e-12
                on_boundary = bm.zeros_like(x, dtype=bool)
                for center in self.centers:
                    cx, cy = center
                    on_boundary |= (x - cx)**2 + (y - cy)**2 < radius**2 + atol
                return on_boundary
            
            @cartesian
            def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                inlet = self.is_inlet_boundary(p)
                wall = self.is_wall_boundary(p)
                top_or_bottom = self.is_top_or_bottom(p)
                obstacle = self.is_obstacle_boundary(p)
                return inlet | wall | top_or_bottom | obstacle

            @cartesian
            def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
                """Check if point where pressure is defined is on boundary."""
                if p is None:
                    return 1
                return self.is_outlet_boundary(p)
        
            @cartesian
            def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
                """Optional: prescribed velocity on boundary, if needed explicitly."""
                inlet = self.inlet_velocity(p)
                is_inlet = self.is_inlet_boundary(p)
            
                result = bm.zeros_like(p, dtype=p.dtype)
                result[is_inlet] = inlet[is_inlet]
                
                return result
            
            @cartesian
            def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
                """Optional: prescribed pressure on boundary (usually for stability)."""
                return self.outlet_pressure(p)
            
            def is_lateral_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
                """Check if point is on boundary."""
                atol = 1e-12
                v0 = p[:, None, :-1] - bd[None, 0::2, :] # (NN, NI, 2)
                v1 = p[:, None, :-1] - bd[None, 1::2, :] # (NN, NI, 2)

                cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
                dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
                cond = (bm.abs(cross) < atol) & (dot < atol)
                return bm.any(cond, axis=1)
            
        model = DLD3D(**options)
        return tuple(
            getattr(model, name)
            for name in ["velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )


from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["IncompressibleNS2d", "IncompressibleCylinder2d"]

class IncompressibleNS2d(CNodeType):
    r"""2D unsteady incompressible Navier-Stokes equations problem model.

    Inputs:
        example (int): Example number.
    
    Outputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        domain (domain): Computational domain.
        velocity (function): Exact velocity solution.
        pressure (function): Exact pressure solution.
        source (function): Source term.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function for velocity boundary regions.
        is_pressure_boundary (function): Predicate function for pressure boundary regions.
    """
    TITLE: str = "二维非稳态不可压缩 NS 方程问题模型"
    PATH: str = "模型.非稳态 NS"
    INPUT_SLOTS = [
        PortConf("example", DataType.MENU, 0, title="例子编号", default=1, items=[i for i in range(1, 3)])
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("domain", DataType.DOMAIN, title="求解域"),
        PortConf("velocity", DataType.FUNCTION, title="速度真解"),
        PortConf("pressure", DataType.FUNCTION, title="压力真解"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]

    @staticmethod
    def run(example) -> Union[object]:
        from fealpy.cfd.model import CFDTestModelManager

        manager = CFDTestModelManager('incompressible_navier_stokes')
        model = manager.get_example(example)
        return (model.mu, model.rho, model.domain()) + tuple(
            getattr(model, name)
            for name in ["velocity", "pressure", "source", "velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )
    

class IncompressibleCylinder2d(CNodeType):
    r"""2D unsteady incompressible Navier-Stokes equations model for flow around a cylinder.

    Inputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        cx (float): x-coordinate of the cylinder center.
        cy (float): y-coordinate of the cylinder center.
        radius (float): Radius of the cylinder.
        n_circle (int): Number of discretization points on the circle.
        h (float): Mesh size.
    
    Outputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        domain (domain): Computational domain.
        source (function): Source term.
        velocity_0 (function): Initial velocity field.
        pressure_0 (function): Initial pressure field.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function for velocity boundary regions.
        is_pressure_boundary (function): Predicate function for pressure boundary regions.
        mesh (mesh): Computational mesh.
    """
    TITLE: str = "二维非稳态圆柱绕流问题模型"
    PATH: str = "模型.非稳态 NS"
    INPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, 0, title="粘度", default=0.001),
        PortConf("rho", DataType.FLOAT, 0, title="密度", default=1.0),
        PortConf("cx", DataType.FLOAT, 0, title="圆心x坐标", default=0.2),
        PortConf("cy", DataType.FLOAT, 0, title="圆心y坐标", default=0.2),
        PortConf("radius", DataType.FLOAT, 0, title="半径", default=0.05),
        PortConf("n_circle", DataType.INT, 0, title="圆离散点数", default=200),
        PortConf("h", DataType.FLOAT, 0, title="网格尺寸", default=0.05)
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="粘度"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("domain", DataType.DOMAIN, title="求解域"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("velocity_0", DataType.FUNCTION, title="初始速度"),
        PortConf("pressure_0", DataType.FUNCTION, title="初始压力"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界"),
        PortConf("mesh", DataType.MESH, title = "网格")
    ]

    @staticmethod
    def run(mu, rho, cx, cy, radius, n_circle, h) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        from fealpy.backend import TensorLike
        from typing import Sequence
        class PDE:
            def __init__(self, options: dict = None):
                self.options = options
                self.atol = 1e-10
                self.mu = options.get('mu', 0.001)
                self.rho = options.get('rho', 1.0)
                self.box = options.get('box', [0.0, 2.2, 0.0, 0.41])
                self.cx = options.get('cx', 0.2)
                self.cy = options.get('cy', 0.2)
                self.radius = options.get('radius', 0.05)
                self.n_circle = options.get('n_circle', 100)
                self.h = options.get('h', 0.01)
                self.center = (self.cx, self.cy)
                self.mesh = self.init_mesh()

            def get_dimension(self) -> int: 
                """Return the geometric dimension of the domain."""
                return 2

            def domain(self) -> Sequence[float]:
                """Return the computational domain [xmin, xmax, ymin, ymax]."""
                return self.box
            
            def init_mesh(self):
                box = self.box 
                center = self.center 
                radius = self.radius 
                n_circle = self.n_circle 
                h = self.h 

                from meshpy.triangle import MeshInfo, build
                from fealpy.mesh import TriangleMesh    
        
                points = [
                    (box[0], box[2]),
                    (box[1], box[2]),
                    (box[1], box[3]),
                    (box[0], box[3])
                ]

                facets = [[0, 1], [1, 2], [2, 3], [3, 0]]

                cx, cy = center
                theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=True)
                circle_points = [(cx + radius*bm.cos(t), cy + radius*bm.sin(t)) for t in theta]
                circle_facets = [[i, (i+1) % n_circle] for i in range(n_circle)]

                circle_offset = len(points)
                all_points = points + circle_points
                all_facets = facets + [[i[0]+circle_offset, i[1]+circle_offset] for i in circle_facets]

                hole_point = [cx, cy]

                mesh_info = MeshInfo()
                mesh_info.set_points(all_points)
                mesh_info.set_facets(all_facets)
                mesh_info.set_holes([hole_point])  # 空洞位置

                mesh = build(mesh_info, max_volume=h**2)

                node = bm.array(mesh.points)
                cell = bm.array(mesh.elements)

                return TriangleMesh(node, cell)
            
            @cartesian
            def velocity_dirichlet(self, p:TensorLike, t) -> TensorLike:
                inlet = self.inlet_velocity(p)
                outlet = self.inlet_velocity(p)
                is_inlet = self.is_inlet_boundary(p)
                is_outlet = self.is_outlet_boundary(p)
                
                result = bm.zeros_like(p, dtype=p.dtype)
                result[is_inlet] = inlet[is_inlet]
                result[is_outlet] = outlet[is_outlet]
                return result
            
            @cartesian
            def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
                return self.outlet_pressure(p)

            @cartesian
            def inlet_velocity(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] =  6 * 1/(0.41)**2 * (0.41 - y) * y
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def outlet_pressure(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of pressure."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape[0], dtype=bm.float64)
                return result
            
            @cartesian
            def wall_velocity(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of velocity on wall."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = bm.array(0.0)
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def wall_pressure(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of pressure on wall."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape[0], dtype=p.dtype)
                result[:] = bm.array(0.0)
                return result
            
            @cartesian
            def obstacle_velocity(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of velocity on obstacle."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = bm.array(0.0)
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def velocity_0(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                return result
            
            @cartesian
            def pressure_0(self, p: TensorLike, t) -> TensorLike:
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape[0], dtype=p.dtype)
                return result
            
            @cartesian
            def source(self, p: TensorLike, t) -> TensorLike:
                """Compute exact source """
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = bm.array(0.0)
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def is_velocity_boundary(self, p):
                return None
            
            @cartesian
            def is_pressure_boundary(self, p : TensorLike = None) -> TensorLike:
                return 0
            
            @cartesian
            def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                return bm.abs(p[..., 0]) < self.atol

            @cartesian
            def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where pressure is defined is on boundary."""
                return bm.abs(p[..., 0] - 2.2) < self.atol
            
            @cartesian
            def is_wall_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                return (bm.abs(p[..., 1] -0.41) < self.atol) | (bm.abs(p[..., 1] ) < self.atol)
            
            @cartesian
            def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                x = p[...,0]
                y = p[...,1]
                cx = self.cx
                cy = self.cy
                r = self.radius
                return (bm.sqrt((x-cx)**2 + (y-cy)**2) - r) < self.atol
        
        options = {
            "mu" : mu,
            "rho" : rho,
            "cx" : cx,
            "cy" : cy,
            "radius" : radius,
            "n_circle" : n_circle,
            "h" : h
        }
        model = PDE(options)
        return (model.mu, model.rho, model.box) + tuple(
            getattr(model, name)
            for name in ["source", "velocity_0", "pressure_0", "velocity_dirichlet", "pressure_dirichlet",
                          "is_velocity_boundary", "is_pressure_boundary", "mesh"]
        )
    
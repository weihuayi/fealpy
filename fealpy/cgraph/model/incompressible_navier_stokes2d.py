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
    DESC: str = """该节点定义二维非稳态不可压缩Navier-Stokes方程模型,可按示例编号加载典型算例，输出
                物性参数、解析解、源项及边界条件，用于数值求解验证。"""
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
    DESC: str = """该节点建立二维非稳态不可压缩圆柱绕流数值模型，自动生成带圆柱障碍物的计算网格，定义
                入口抛物线速度、出口压力及各类边界条件, 为Navier-Stokes方程求解提供完整物理场输入。"""
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
                import gmsh 
                from fealpy.mesh import TriangleMesh 
                box = self.box 
                center = self.center 
                radius = self.radius 
                n_circle = self.n_circle 
                h = self.h 
                cx = self.center[0]
                cy = self.center[1] 
                gmsh.initialize() 
                gmsh.model.add("rectangle_with_polygon_hole") 
                xmin, xmax, ymin, ymax = box 
                p1 = gmsh.model.geo.addPoint(xmin, ymin, 0) 
                p2 = gmsh.model.geo.addPoint(xmax, ymin, 0) 
                p3 = gmsh.model.geo.addPoint(xmax, ymax, 0) 
                p4 = gmsh.model.geo.addPoint(xmin, ymax, 0) 
                l1 = gmsh.model.geo.addLine(p1, p2) 
                l2 = gmsh.model.geo.addLine(p2, p3) 
                l3 = gmsh.model.geo.addLine(p3, p4) 
                l4 = gmsh.model.geo.addLine(p4, p1) 
                outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4]) 
                theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False) 
                circle_pts = [] 
                for t in theta:
                    x = cx + radius * bm.cos(t) 
                    y = cy + radius * bm.sin(t) 
                    pid = gmsh.model.geo.addPoint(x, y, 0) 
                    circle_pts.append(pid) 
                circle_lines = [] 
                for i in range(n_circle): 
                    l = gmsh.model.geo.addLine(circle_pts[i], circle_pts[(i + 1) % n_circle]) 
                    circle_lines.append(l) 
                circle_loop = gmsh.model.geo.addCurveLoop(circle_lines) 
                surf = gmsh.model.geo.addPlaneSurface([outer_loop, circle_loop]) 
                gmsh.model.geo.synchronize() 
                inlet = gmsh.model.addPhysicalGroup(1, [l4], tag = 1) 
                gmsh.model.setPhysicalName(1, 1, "inlet") 
                outlet = gmsh.model.addPhysicalGroup(1, [l2], tag = 2) 
                gmsh.model.setPhysicalName(1, 2, "outlet") 
                wall = gmsh.model.addPhysicalGroup(1, [l1, l3], tag = 3) 
                gmsh.model.setPhysicalName(1, 3, "walls") 
                cyl = gmsh.model.addPhysicalGroup(1, circle_lines, tag = 4) 
                gmsh.model.setPhysicalName(1, 4, "cylinder") 
                domain = gmsh.model.addPhysicalGroup(2, [surf], tag = 5) 
                gmsh.model.setPhysicalName(2, 5, "fluid") 
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h) 
                gmsh.model.mesh.generate(2) 
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes() 
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2) 
                tri_nodes = elem_node_tags[0].reshape(-1, 3) - 1 # 转为从0开始索引 
                node_coords = bm.array(node_coords).reshape(-1, 3)[:, :2] 
                tri_nodes = bm.array(tri_nodes, dtype=bm.int32) 
                boundary = [] 
                boundary_tags = [1, 2, 3, 4] 
                for tag in boundary_tags: 
                    node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag) # 转换为从 0 开始的索引 
                    boundary.append(bm.array(node_tags - 1, dtype=bm.int32)) 
                self.boundary = boundary 
                gmsh.finalize() 
                return TriangleMesh(node_coords, tri_nodes)
            
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
    

class FlowPastFoil(CNodeType):
    TITLE: str = "二维翼型绕流问题模型"
    PATH: str = "模型.非稳态 NS"
    DESC: str = """该节点建立二维非稳态翼型绕流数值模型,自动生成带NACA001翼型障碍物的计算网格, 定义
                入口抛物线速度、出口压力及各类边界条件, 为Navier-Stokes方程求解提供完整物理场输入。"""
    INPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, 0, title="粘度", default=0.001),
        PortConf("rho", DataType.FLOAT, 0, title="密度", default=1.0),
        PortConf("inflow", DataType.FLOAT, 0, title="入口最大速度", default=1.0),
        PortConf("box", DataType.TENSOR, 1, title="求解域"),
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
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]

    @staticmethod
    def run(mu, rho, inflow, box) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian, TensorLike
        class PDE():
            def __init__(self, options : dict = None ):
                self.eps = 1e-10
                self.options = options
                self.rho = options.get('rho', 1.0)
                self.mu = options.get('mu', 0.001)
                self.box = options.get('box', [-0.5, 2.7, -0.5, 0.5])
                self.inflow = options.get('inflow', 1.0)

            @cartesian
            def is_outflow_boundary(self,p):
                x = p[...,0]
                y = p[...,1]
                cond1 = bm.abs(x - self.box[1]) < self.eps
                cond2 = bm.abs(y-self.box[2])>self.eps
                cond3 = bm.abs(y-self.box[3])>self.eps
                return (cond1) & (cond2 & cond3) 
            
            @cartesian
            def is_inflow_boundary(self,p):
                return bm.abs(p[..., 0]-self.box[0]) < self.eps
            
            @cartesian
            def is_wall_boundary(self,p):
                return (bm.abs(p[..., 1] -self.box[2]) < self.eps) | \
                    (bm.abs(p[..., 1] -self.box[3]) < self.eps)
            
            @cartesian
            def is_velocity_boundary(self,p):
                return ~self.is_outflow_boundary(p)
                # return None
            
            @cartesian
            def is_pressure_boundary(self,p=None):
                if p is None:
                    return 1
                else:
                    return self.is_outflow_boundary(p) 
                    #return bm.zeros_like(p[...,0], dtype=bm.bool)
                # return 0

            @cartesian
            def u_inflow_dirichlet(self, p):
                x = p[...,0]
                y = p[...,1]
                value = bm.zeros_like(p)
                # value[...,0] = 1.5 * 4 * (y-self.box[2])*(self.box[3]-y)/((0.41)**2)
                value[...,0] = self.inflow
                # value[...,1] = 0
                return value
            
            @cartesian
            def pressure_dirichlet(self, p, t):
                x = p[...,0]
                y = p[...,1]
                value = bm.zeros_like(x)
                return value

            @cartesian
            def velocity_dirichlet(self, p, t):
                x = p[...,0]
                y = p[...,1]
                index = self.is_inflow_boundary(p)
                # outlet = self.is_outflow_boundary(p)
                result = bm.zeros_like(p)
                result[index] = self.u_inflow_dirichlet(p[index])
                # result[outlet] = self.u_inflow_dirichlet(p[outlet])
                return result
            
            @cartesian
            def source(self, p, t):
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = 0
                result[..., 1] = 0
                return result
            
            def velocity_0(self, p ,t):
                x = p[...,0]
                y = p[...,1]
                value = bm.zeros(p.shape)
                return value

            def pressure_0(self, p, t):
                x = p[..., 0]
                val = bm.zeros_like(x)
                return val
            
        options ={
            'rho': rho,
            'mu': mu,
            'inflow': inflow,
            'box': box,
        }
        model = PDE(options)

        return (model.mu, model.rho, model.box) + tuple(
            getattr(model, name)
            for name in ["source", "velocity_0", "pressure_0", "velocity_dirichlet", "pressure_dirichlet",
                          "is_velocity_boundary", "is_pressure_boundary"]
        )
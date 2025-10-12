from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["IncompressibleNS2d", "IncompressibleCylinder2d"]

class IncompressibleNS2d(CNodeType):
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
    TITLE: str = "二维非稳态圆柱绕流问题模型"
    PATH: str = "模型.非稳态 NS"
    INPUT_SLOTS = [
        PortConf("n_circle", DataType.FLOAT, 0, title="圆柱分段数"),
        PortConf("h", DataType.FLOAT, 0, max_val=0.1 ,title="网格密度")
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
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
    def run(n_circle, h) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        from fealpy.backend import TensorLike
        from typing import Sequence
        class PDE:
            def __init__(self, options: dict = None):
                self.options = options
                self.atol = 1e-10
                self.mu = 0.001
                self.rho = 1.0

                if options is not None:
                    self.box = options.get('box', [0.0, 2.2, 0.0, 0.41])
                    self.center = options.get('center', (0.2, 0.2))
                    self.radius = options.get('radius', 0.05)
                    self.n_circle = options.get('n_circle', 100)
                    self.h = options.get('h', 0.05)
                self.mesh = self.init_mesh()

            def get_dimension(self) -> int: 
                """Return the geometric dimension of the domain."""
                return 2

            def domain(self) -> Sequence[float]:
                """Return the computational domain [xmin, xmax, ymin, ymax]."""
                return self.box
            
            def init_mesh(self, box = [0.0, 2.2, 0.0, 0.41], center = (0.2, 0.2), radius = 0.05, n_circle = 100, h = 0.01):
                
                box = self.box if self.options is not None else box
                center = self.center if self.options is not None else center
                radius = self.radius if self.options is not None else radius
                n_circle = self.n_circle if self.options is not None else n_circle
                h = self.h if self.options is not None else h

                from meshpy.triangle import MeshInfo, build
                from fealpy.mesh import TriangleMesh    
                # 矩形顶点
                points = [
                    (box[0], box[2]),
                    (box[1], box[2]),
                    (box[1], box[3]),
                    (box[0], box[3])
                ]

                # 矩形边界
                facets = [[0, 1], [1, 2], [2, 3], [3, 0]]

                # 圆的离散点
                cx, cy = center
                theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=True)
                circle_points = [(cx + radius*bm.cos(t), cy + radius*bm.sin(t)) for t in theta]

                # 圆的边界：顺时针编号（meshpy 要求空洞边界为顺时针）
                circle_facets = [[i, (i+1) % n_circle] for i in range(n_circle)]

                # 合并点和边界
                circle_offset = len(points)
                all_points = points + circle_points
                all_facets = facets + [[i[0]+circle_offset, i[1]+circle_offset] for i in circle_facets]

                # 设置空洞区域（空洞内一点）
                hole_point = [cx, cy]

                # meshpy 生成三角网格
                mesh_info = MeshInfo()
                mesh_info.set_points(all_points)
                mesh_info.set_facets(all_facets)
                mesh_info.set_holes([hole_point])  # 空洞位置

                mesh = build(mesh_info, max_volume=h**2)

                node = bm.array(mesh.points)
                cell = bm.array(mesh.elements)

                # 转为 FEALPy 的 TriangleMesh
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
                pi = bm.pi
                sin = bm.sin
                result = bm.zeros(p.shape, dtype=bm.float64)
                # result[..., 0] = sin(pi*t/8) * 1/(0.41)**2 * (0.41 - y) * 6 * y
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
                # is_out = self.is_outlet_boundary(p)
                # return ~is_out
                # inlet = self.is_inlet_boundary(p)
                # wall = self.is_wall_boundary(p)
                # obstacle = self.is_obstacle_boundary(p)
                # return inlet|wall|obstacle
                return None
            
            @cartesian
            def is_pressure_boundary(self, p : TensorLike = None) -> TensorLike:
                # if p is None:
                #     return 1
                # is_out = self.is_outlet_boundary(p)
                # return is_out
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
                return (bm.sqrt((x-0.2)**2 + (y-0.2)**2) - 0.05) < self.atol
        
        options = {
            "n_circle" : n_circle,
            "lc" : h
        }
        model = PDE(options)
        return (model.mu, model.rho, model.domain()) + tuple(
            getattr(model, name)
            for name in ["source", "velocity_0", "pressure_0", "velocity_dirichlet", "pressure_dirichlet",
                          "is_velocity_boundary", "is_pressure_boundary", "mesh"]
        )
    
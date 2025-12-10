from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

class RayleighTaylor(CNodeType):
    r"""Rayleigh–Taylor Instability (RTI) problem setup for CHNS equation.

    This node defines the physical parameters, computational domain,
    and boundary/initial conditions for the Rayleigh–Taylor instability
    problem governed by the Cahn–Hilliard–Navier–Stokes (CHNS) equations.

    Inputs:
        rho_up (float): Density of the upper fluid.
        rho_down (float): Density of the lower fluid.
        Re (float): Reynolds number.
        Fr (float): Froude number.
        epsilon (float): Interface thickness parameter.
        eps (float): Tolerance for boundary comparison.

    Outputs:
        rho_up (float): Density of the upper fluid.
        rho_down (float): Density of the lower fluid.
        Re (float): Reynolds number.
        Fr (float): Froude number.
        epsilon (float): Interface thickness parameter.
        Pe (float): Peclet number, computed as 1/epsilon.
        domain (list): Computational domain of the problem.
        velocity_boundary (Callable): Velocity boundary condition function.
        pressure_boundary (Callable): Pressure boundary condition function.
        is_u_boundary (Callable): Predicate for velocity boundary points.
        is_ux_boundary (Callable): Predicate for velocity-x boundary points.
        is_uy_boundary (Callable): Predicate for velocity-y boundary points.
        is_pressure_boundary (Callable): Predicate for pressure boundary points.
        init_interface (Callable): Initial phase interface function.
    """

    TITLE: str = "Rayleigh-Taylor Instability(RTI)问题模型"
    PATH: str = "模型.CHNS 方程"
    DESC: str = """该节点定义了 Cahn–Hilliard–Navier–Stokes (CHNS) 方程中 Rayleigh–Taylor 
                不稳定性问题的物理参数、计算域以及边界与初始条件。通过输入上层与下层流体的密度、雷诺数、
                弗劳德数、界面厚度参数等物理参数，自动构造计算域及相应的边界条件函数与初始界面函数，便
                于后续数值求解过程的设置与调用。
                使用示例：传入所需物理参数，连接该节点后即可获取问题的各项设置，用于后续的网格生成、函
                数空间定义及数值求解器配置。
                """
    INPUT_SLOTS = [
        PortConf("rho_up", DataType.FLOAT, 0, title="上层流体密度", default=3.0),
        PortConf("rho_down", DataType.FLOAT, 0, title="下层流体密度", default=1.0),
        PortConf("Re", DataType.FLOAT, 0, title="雷诺数", default=3000.0),
        PortConf("Fr", DataType.FLOAT, 0, title="弗劳德数", default=1.0),
        PortConf("epsilon", DataType.FLOAT, 0, title="界面厚度参数", default=0.01),
        PortConf("eps", DataType.FLOAT, 0, title="残差", default=1e-10)
    ]
    OUTPUT_SLOTS = [
        PortConf("rho_up", DataType.FLOAT, title="上层流体密度"),
        PortConf("rho_down", DataType.FLOAT, title="下层流体密度"),
        PortConf("Re", DataType.FLOAT, title="雷诺数"),
        PortConf("Fr", DataType.FLOAT, title="弗劳德数"),
        PortConf("epsilon", DataType.FLOAT, title="界面厚度参数"),
        PortConf("Pe", DataType.FLOAT, title="Peclet 数"),
        PortConf("domain", DataType.FUNCTION, title="计算域"),
        PortConf("velocity_boundary", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_boundary", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_u_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_ux_boundary", DataType.FUNCTION, title="速度 x 分量边界"),
        PortConf("is_uy_boundary", DataType.FUNCTION, title="速度 y 分量边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界"),
        PortConf("init_interface", DataType.FUNCTION, title="初始界面函数")
    ]

    @staticmethod
    def run(**options) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        from fealpy.mesh import TriangleMesh
        class RayleignTaylor:
            def __init__(self, options: dict = {}):
                self.rho_up = options.get('rho_up', 3)
                self.rho_down = options.get('rho_down', 1)
                self.Re = options.get('Re', 3000)
                self.Fr = options.get('Fr', 1)
                self.epsilon = options.get('epsilon', 0.01)
                self.Pe = 1/self.epsilon
                self.eps = options.get('eps', 1e-10)    
                self.domain = [0.0, 1.0, 0.0, 4.0]

            def init_mesh(self, nx=128, ny=512):
                '''
                生成网格
                nx, ny: 网格数目
                '''
                mesh = TriangleMesh.from_box(self.domain, nx, ny)
                self.mesh = mesh
                return mesh
            
            @cartesian
            def init_interface(self, p):
                '''
                初始化界面
                '''
                x = p[...,0]
                y = p[...,1]
                val =  bm.tanh((y-2-0.1*bm.cos(bm.pi*2*x))/(bm.sqrt(bm.tensor(2))*self.epsilon))
                return val
            
            @cartesian
            def velocity_boundary(self, p):
                '''
                边界速度
                '''
                result = bm.zeros_like(p, dtype=bm.float64)
                return result
            
            @cartesian
            def pressure_boundary(self, p):
                '''
                边界压力
                '''
                result = bm.zeros_like(p[..., 0], dtype=bm.float64)
                return result

            @cartesian
            def is_p_boundary(self, p):
                result = bm.zeros_like(p[..., 0], dtype=bool)
                return result

            @cartesian
            def is_u_boundary(self, p):
                tag_up = bm.abs(p[..., 1] - 4.0) < self.eps
                tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
                tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
                tag_right = bm.abs(p[..., 0] - 1) < self.eps
                return tag_up | tag_down | tag_left | tag_right 
            
            @cartesian
            def is_ux_boundary(self, p):
                tag_up = bm.abs(p[..., 1] - 4.0) < self.eps
                tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
                tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
                tag_right = bm.abs(p[..., 0] - 1) < self.eps
                return tag_up | tag_down | tag_left | tag_right 
            
            @cartesian
            def is_uy_boundary(self, p):
                tag_up = bm.abs(p[..., 1] - 4.0) < self.eps
                tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
                return tag_up | tag_down

            @cartesian
            def is_pressure_boundary(self):
                return 0
            
        model = RayleignTaylor(options)


        return (model.rho_up, model.rho_down, model.Re, model.Fr,
                model.epsilon, model.Pe, model.domain,
                model.velocity_boundary, model.pressure_boundary,
                model.is_u_boundary, model.is_ux_boundary, model.is_uy_boundary,
                model.is_pressure_boundary, model.init_interface)
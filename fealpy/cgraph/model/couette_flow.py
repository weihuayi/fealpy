from typing import Union
from ..nodetype import CNodeType, PortConf, DataType


class CouetteFlow(CNodeType):
    r"""Two-Phase Couette Flow Model (Phase-Field Method)

    This node provides a complete setup for the classic two-phase Couette flow benchmark:
    - Rectangular domain: [-0.5, 0.5] × [-0.125, 0.125]
    - Top wall (y = +0.125) moves right with velocity +0.2
    - Bottom wall (y = -0.125) moves left with velocity -0.2
    - Initial flat interface separating two immiscible fluids (φ = +1 and φ = -1)

    All dimensionless physical parameters, mesh generation, initial phase field,
    boundary identification functions, and recommended time steps are provided.
    Ideal for testing and validating phase-field-based two-phase incompressible
    Navier-Stokes solvers with moving contact lines and generalized Navier
    boundary conditions (slip + contact angle).

    Inputs
    ------
    eps   : Tolerance for geometric boundary detection
    T     : Total simulation time
    h     : Mesh element size (spatial resolution)
    theta : Static contact angle in degrees (converted to radians internally)

    Outputs
    -------
    param_list : List of 9 key dimensionless parameters in fixed order:
                 [R, L_s, epsilon, L_d, lam, V_s, s, theta_s(rad), T]
    init_phi   : Initial phase-field distribution function
    Boundary markers and velocity BC functions
    domain, nt, h, nx, ny for downstream solver setup
    """
    TITLE: str = "两相Couette流动问题模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点定义了两相Couette流动的物理模型、计算域以及边界与初始条件。
        模型中上下两壁分别以相反方向移动，从而在两相流体间产生剪切流动。
        同时定义了相场厚度、滑移长度、势能系数、壁面速度等无量纲参数，
        用于验证相场法与两相流数值算法的正确性与稳定性。"""
    INPUT_SLOTS = [
        PortConf("eps", DataType.FLOAT, 0, title="边界判断精度", default=1e-10),
        PortConf("T", DataType.FLOAT, 0, title="总时间进程", default=2),
        PortConf("h", DataType.FLOAT, 0, title="网格尺寸", default=1/256),
        PortConf("theta", DataType.FLOAT, 0, title="接触角", default=77.6)
    ]
    OUTPUT_SLOTS = [
        PortConf("param_list", DataType.LIST, title="参数列表"),
        PortConf("init_phi", DataType.FUNCTION, title="定义初始相场分布"),
        PortConf("is_uy_Dirichlet", DataType.FUNCTION, title="判断是否为速度Dirichlet边界"),
        PortConf("is_up_boundary", DataType.FUNCTION, title="判断是否为上边界"),
        PortConf("is_down_boundary", DataType.FUNCTION, title="判断是否为下边界"),
        PortConf("is_wall_boundary", DataType.FUNCTION, title="判断是否为壁面边界"),
        PortConf("u_w", DataType.FUNCTION, title="定义壁面速度边界条件"),
        PortConf("domain", DataType.NONE, title = "求解区域"),
        PortConf("dt", DataType.FLOAT, title="时间步长"),
        PortConf("nt", DataType.INT, title="最大迭代次数"),
        PortConf("h", DataType.FLOAT, 0, title="网格尺寸", default=1/256),
        PortConf("nx", DataType.INT, title="x方向单元数"),
        PortConf("ny", DataType.INT, title="y方向单元数"),
    ]

    @staticmethod
    def run(eps, T, h, theta) -> Union[object]:
        from fealpy.decorator import cartesian
        from fealpy.backend import backend_manager as bm
        bm.set_backend("pytorch")
        class PDE:
            def __init__(self, eps=1e-10,T = 2,h=1/256,theta = 77.6):
                self.eps = eps
                
                ## init the parameter
                self.R = 5.0 ##dimensionless
                self.l_s = 0.0025 ##dimensionless slip length
                self.L_s = self.l_s / 1000
                self.epsilon = 0.004 ## the thickness of interface
                self.L_d = 0.0005 ##phenomenological mobility cofficient
                self.lam = 12.0 ##dimensionless
                self.V_s = 200.0 ##dimensionless 
                self.s = 2.5 ##stablilizing parameter
                self.domain = [-0.5, 0.5, -0.125, 0.125]
                self.theta_s = bm.array(theta/180 * bm.pi)
                self.T = T
                self.nt = int(T/(0.1*h))
                self.dt = T/self.nt
                self.h = h
                self.nx = int(1/self.h)
                self.ny = int(0.25/self.h)
                
                
            @cartesian
            def is_wall_boundary(self,p):
                return (bm.abs(p[..., 1] - 0.125) < self.eps) | \
                    (bm.abs(p[..., 1] + 0.125) < self.eps)
            
            @cartesian
            def is_up_boundary(self,p):
                return bm.abs(p[..., 1] - 0.125) < self.eps
            
            @cartesian
            def is_down_boundary(self,p):
                return bm.abs(p[..., 1] + 0.125) < self.eps
            
            @cartesian
            def is_uy_Dirichlet(self,p):
                return (bm.abs(p[..., 1] - 0.125) < self.eps) | \
                    (bm.abs(p[..., 1] + 0.125) < self.eps)
            
            @cartesian
            def init_phi(self,p):
                x = p[..., 0]
                y = p[..., 1]   
                tagfluid0 = bm.logical_and(x > -0.25, x < 0.25)
                tagfluid1 = bm.logical_not(tagfluid0)
                phi = bm.zeros_like(x)
                phi[tagfluid0] = 1.0
                phi[tagfluid1] = -1.0
                return phi

            @cartesian        
            def u_w(self, p):
                y = p[..., 1]
                result = bm.zeros_like(p)
                tag_up = (bm.abs(y-0.125)) < self.eps 
                tag_down = (bm.abs(y+0.125)) < self.eps
                value = bm.where(tag_down, -0.2, 0) + bm.where(tag_up, 0.2, 0)
                result[..., 0] = value 
                return result

            @cartesian
            def p_dirichlet(self, p):
                return bm.zeros_like(p[..., 0])

            @cartesian
            def is_p_dirichlet(self, p):
                return bm.zeros_like(p[..., 0], dtype=bool)
            
        options = {
            "eps" : eps,
            "T" : T,
            "h" : h,
            "theta" : theta
        }
        model = PDE(eps=options["eps"], h=options["h"])
        param_list = [
        model.R,      
        model.L_s,     
        model.epsilon, 
        model.L_d,     
        model.lam,     
        model.V_s,     
        model.s,       
        model.theta_s,
        ]
        return (param_list,model.init_phi, model.is_uy_Dirichlet,
                model.is_up_boundary, model.is_down_boundary,
                model.is_wall_boundary, model.u_w, model.domain,
                model.dt,model.nt,model.h,model.nx,model.ny
                )
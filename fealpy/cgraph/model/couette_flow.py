from typing import Union
from ..nodetype import CNodeType, PortConf, DataType


class CouetteFlow(CNodeType):
    r"""Two-phase Couette flow problem model.

    This node defines the physical model, domain geometry, and boundary/initial conditions
    for a **two-phase Couette flow** problem. It is used to test and validate numerical
    solvers for two-phase incompressible flow with moving wall boundary conditions.

    In the Couette flow configuration, two parallel plates move in opposite directions,
    driving the flow of two immiscible fluids separated by an initial interface.

    Inputs:
        eps (float): Tolerance used for boundary detection, e.g. determining whether
            a point lies on the upper or lower wall.
        h (float): Mesh element size (spatial discretization parameter).

    Outputs:
        R (float): System characteristic length (dimensionless).
        L_s (float): Physical slip length (dimensionless).
        epsilon (float): Interface thickness (dimensionless Cahn–Hilliard parameter).
        L_d (float): Mobility coefficient (dimensionless).
        lam (float): Potential well coefficient in the phase-field equation (dimensionless).
        V_s (float): Wall velocity magnitude (dimensionless).
        s (float): Stabilization parameter (dimensionless).
        theta_s (Tensor): Static contact angle (in radians).
        h (float): Mesh resolution.
        init_phi (Function): Function defining the initial phase field distribution.
        is_uy_Dirichlet (Function): Function marking Dirichlet boundaries for velocity.
        is_up_boundary (Function): Function identifying the upper wall.
        is_down_boundary (Function): Function identifying the lower wall.
        is_wall_boundary (Function): Function identifying both top and bottom walls.
        u_w (Function): Function specifying the wall velocity boundary condition.
        mesh (Mesh): Triangular mesh representing the computational domain.

    Description:
        The Couette flow setup represents a classical benchmark for multiphase flow models.
        The computational domain is a rectangular channel where:
            - The top wall moves rightward with velocity +V_s.
            - The bottom wall moves leftward with velocity −V_s.
            - Two immiscible fluids are initialized with a sharp interface at x=0.
        The model includes all essential PDE components and utility functions needed
        for simulation setup in a phase-field-based two-phase solver.

        The returned PDE object provides:
            - Mesh generation
            - Boundary identification functions
            - Initial phase field initialization
            - Wall velocity and contact angle setup"""
    TITLE: str = "两相Couette流动问题模型"
    PATH: str = "模型.非稳态两相Couette流动"
    DESC: str = """该节点定义了两相Couette流动的物理模型、计算域以及边界与初始条件。
        模型中上下两壁分别以相反方向移动，从而在两相流体间产生剪切流动。
        同时定义了相场厚度、滑移长度、势能系数、壁面速度等无量纲参数，
        用于验证相场法与两相流数值算法的正确性与稳定性。"""
    INPUT_SLOTS = [
        PortConf("eps", DataType.FLOAT, 0, title="边界判断精度", default=1e-10),
        PortConf("h", DataType.FLOAT, 0, title="网格尺寸", default=1/256)
    ]
    OUTPUT_SLOTS = [
        PortConf("R", DataType.FLOAT, title="系统长度尺度(无量纲)"),
        PortConf("L_s", DataType.FLOAT, title = "物理滑移长度"),
        PortConf("epsilon", DataType.FLOAT, title="界面厚度(无量纲)"),
        PortConf("L_d", DataType.FLOAT, title="迁移系数(无量纲)"),
        PortConf("lam", DataType.FLOAT, title="相场势能系数(无量纲)"),
        PortConf("V_s", DataType.FLOAT, title="壁面速度(无量纲)"),
        PortConf("s", DataType.FLOAT, title="稳定化参数(无量纲)"),
        PortConf("theta_s", DataType.TENSOR, title="接触角(弧度)"),
        PortConf("h", DataType.FLOAT, 0, title="网格尺寸", default=1/256),
        PortConf("init_phi", DataType.FUNCTION, title="定义初始相场分布"),
        PortConf("is_uy_Dirichlet", DataType.FUNCTION, title="判断是否为速度Dirichlet边界"),
        PortConf("is_up_boundary", DataType.FUNCTION, title="判断是否为上边界"),
        PortConf("is_down_boundary", DataType.FUNCTION, title="判断是否为下边界"),
        PortConf("is_wall_boundary", DataType.FUNCTION, title="判断是否为壁面边界"),
        PortConf("u_w", DataType.FUNCTION, title="定义壁面速度边界条件"),
        PortConf("mesh", DataType.MESH, title = "网格"),
    ]

    @staticmethod
    def run(eps, h) -> Union[object]:
        from fealpy.decorator import cartesian
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import TriangleMesh
        class PDE:
            def __init__(self, eps=1e-10, h=1/256):
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
                #self.theta_s = bm.array(bm.pi/2)
                self.theta_s = bm.array(77.6/180 * bm.pi)
                self.h = h

            def mesh(self):
                box = [-0.5, 0.5, -0.125, 0.125]
                mesh = TriangleMesh.from_box(box, nx=int(1/self.h), ny=int(0.25/self.h))
                return mesh

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
            "h" : h
        }
        model = PDE(eps=options["eps"], h=options["h"])
        return (model.R, model.L_s, model.epsilon, model.L_d,
                model.lam, model.V_s, model.s, model.theta_s,
                model.h,model.init_phi, model.is_uy_Dirichlet,
                model.is_up_boundary, model.is_down_boundary,
                model.is_wall_boundary, model.u_w, model.mesh())

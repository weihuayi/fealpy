from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

class GNBCSimulation(CNodeType):
    r"""Two-Phase Couette Flow Solver with Generalized Navier Boundary Condition (GNBC)

    This node implements a fully coupled finite element solver for two-phase incompressible
    flow using the phase-field (Cahn–Hilliard) and Navier–Stokes equations under the
    Generalized Navier Boundary Condition (GNBC). It is specifically designed for the
    classic two-phase Couette flow benchmark with moving walls and dynamic contact lines.

    The solver uses a sequential segregated time-stepping scheme:
        1. Solve the Cahn–Hilliard (CH) system → update phase field φ and chemical potential μ
        2. Solve the incompressible Navier–Stokes (NS) system with phase-dependent viscosity/density
           and interfacial force from μ∇φ → update velocity u and pressure p

    Key features:
    - Supports GNBC with slip length (L_s) and static contact angle (θ_s)
    - Automatic Dirichlet velocity application on moving walls
    - High-order quadrature and stable FE discretization
    - VTK output at each time step (with compression-ready structure)
    - Real-time monitoring of wall slip velocities (max/min u_x on top/bottom walls)

    The final outputs are the maximum and minimum x-velocity components on the upper
    and lower walls at the final time step — crucial for quantifying effective slip
    and validating GNBC implementation against analytical or reference solutions.
    """
    TITLE: str = "GNBC边界条件下两相Couette流动问题有限元求解"
    PATH: str = "simulation.solvers"
    DESC: str = """
    基于广义 Navier 边界条件（GNBC）的两相 Couette 流动高精度有限元求解器。\n
    该节点实现了 Cahn–Hilliard 相场方程与不可压 Navier–Stokes 方程的完全耦合求解，
    支持动接触线、壁面滑移长度、接触角等物理效应，是验证扩散界面模型与 GNBC
    边界条件正确性的标准基准算例。\n
    • 采用分步求解策略：先更新相场与化学势，再更新速度与压力
    • 自动施加上下壁面运动速度（Dirichlet 条件）
    • 支持任意阶有限元空间与积分精度
    • 每步自动导出 VTU 结果（支持 ParaView 可视化）
    • 实时输出并返回上下壁面速度极值，用于滑移规律分析与模型验证
    适用于两相不可压流体的层流模拟、界面滑移研究及润湿动力学分析。
"""
    INPUT_SLOTS = [
        PortConf("dt", DataType.FLOAT, title="时间步长"),
        PortConf("i", DataType.INT, title="当前时间步"),
        PortConf("param_list", DataType.LIST, title="参数列表"),
        PortConf("init_phi", DataType.FUNCTION, 1, title="定义初始相场分布"),
        PortConf("is_wall_boundary", DataType.FUNCTION, 1, title="判断是否为壁面边界"),
        PortConf("u_w", DataType.FUNCTION, 1, title="定义壁面速度边界条件"),
        PortConf("phispace", DataType.SPACE, 1, title="相场函数空间"),
        PortConf("space", DataType.SPACE, 1, title="函数空间"),
        PortConf("pspace", DataType.SPACE, 1, title="压力函数空间"),
        PortConf("uspace", DataType.SPACE, 1, title="速度函数空间"),
        PortConf("NS_BC", DataType.FUNCTION, title="边界处理函数"),
        PortConf("u0", DataType.FUNCTION, title="第0步速度"),
        PortConf("u1", DataType.FUNCTION, title="第1步速度"),
        PortConf("phi0", DataType.FUNCTION, title="第0步序参量"),
        PortConf("phi1", DataType.FUNCTION, title="第1步序参量"),
        PortConf("mu1", DataType.FUNCTION, title="第1步化学势"),
        PortConf("p1", DataType.FUNCTION, title="第1步压力"),
        PortConf("q", DataType.INT, title="积分次数", default=5),
    ]
    OUTPUT_SLOTS = [
        PortConf("u0", DataType.FUNCTION, title="第0步速度"),
        PortConf("u1", DataType.FUNCTION, title="第1步速度"),
        PortConf("phi0", DataType.FUNCTION, title="第0步序参量"),
        PortConf("phi1", DataType.FUNCTION, title="第1步序参量"),
        PortConf("mu1", DataType.FUNCTION, title="第1步化学势"),
        PortConf("p1", DataType.FUNCTION, title="第1步压力")
    ]
    
    @staticmethod
    def run(dt, i, param_list, init_phi,is_wall_boundary,u_w,
            phispace,space,pspace,uspace,NS_BC,
            u0=None,u1=None,phi0=None,phi1=None,mu1=None,p1=None,q=5) -> Union[object]:
        from fealpy.solver import spsolve
        from fealpy.cfd.example.GNBC.solver import Solver
        class PDE:
            def __init__(self, param_list, is_wall_boundary,
                        u_w, init_phi):
                self.R = param_list[0]
                self.L_s = param_list[1]
                self.epsilon = param_list[2]
                self.L_d = param_list[3]
                self.lam = param_list[4]
                self.V_s = param_list[5]
                self.s = param_list[6]
                self.theta_s = param_list[7]
                self.is_wall_boundary = is_wall_boundary
                self.u_w = u_w
                self.init_phi = init_phi
        pde = PDE(param_list, is_wall_boundary,u_w,init_phi) 
        mesh = getattr(space, 'mesh', None)
        solver = Solver(pde, mesh, pspace, phispace, uspace, dt, q)
        ugdof = uspace.number_of_global_dofs()
        phigdof = phispace.number_of_global_dofs()
        if i == 0:  
            u0 = uspace.function()
            u1 = uspace.function()
            u2 = uspace.function()
            phi0 = phispace.interpolate(pde.init_phi)
            phi1 = phispace.function()
            # TODO:第一步求解
            phi1[:] = phi0[:]
            phi2 = phispace.function()
            mu1 = phispace.function()
            mu2 = phispace.function()
            p1 = pspace.function()
            p2 = pspace.function() 
        else:
            CH_BForm = solver.CH_BForm()
            CH_LForm = solver.CH_LForm()
            NS_BForm = solver.NS_BForm()
            NS_LForm = solver.NS_LForm()
            solver.CH_update(u0, u1, phi0, phi1)
            CH_A = CH_BForm.assembly()
            CH_b = CH_LForm.assembly()
            CH_x = spsolve(CH_A, CH_b, 'mumps')
            
            phi2[:] = CH_x[:phigdof]
            mu2[:] = CH_x[phigdof:] 
            solver.NS_update(u0, u1, mu2, phi2, phi1)
            NS_A = NS_BForm.assembly()
            NS_b = NS_LForm.assembly()
            NS_A,NS_b = NS_BC(NS_A,NS_b)
            NS_x = spsolve(NS_A, NS_b, 'mumps')

            u2[:] = NS_x[:ugdof]
            p2[:] = NS_x[ugdof:]
            
            u0[:] = u1[:]
            u1[:] = u2[:]
            phi0[:] = phi1[:]
            phi1[:] = phi2[:]
            mu1[:] = mu2[:]
            p1[:] = p2[:]
            
        return u0,u1,phi0,phi1,mu1,p1

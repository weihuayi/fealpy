from ..nodetype import CNodeType, PortConf, DataType

class CahnHilliardFEMSimulation(CNodeType):
    r"""Finite Element Discretization for the Cahn-Hilliard-Navier-Stokes (CHNS) Equation.

    This node constructs the finite element bilinear and linear forms
    required for the discretization of the Cahn-Hilliard (phase-field) subsystem
    in the CHNS model. The method assembles a block system involving the order 
    parameter (φ) and chemical potential (μ) using the provided function space.

    Inputs:
        epsilon (float): Interface thickness parameter.
        Pe (float): Peclet number.
        phispace (SpaceType): Finite element space for the phase field φ.
        q (int): Quadrature degree used for numerical integration.
        s (float): Stabilization parameter.

    Outputs:
        update (Callable): Function that assembles the bilinear and linear forms
                           (A, L) for each time step based on current and previous fields.
    """
    TITLE: str = "CHNS 方程有限元离散"
    PATH: str = "流体.有限元算法"
    DESC: str = """
                使用有限元方法对 Cahn–Hilliard–Navier–Stokes (CHNS) 方程中的相场部分进行离散。
                本节点实现了相场方程的有限元双线性形式与线性形式组装，适用于相场变量 φ 与化学势 μ 的耦
                合系统。通过输入界面厚度参数、Peclet 数、相场函数空间 、积分精度以及稳定参数，自动构造
                相应的双线性形式 (A00, A01, A10, A11) 与线性形式 (L0, L1)，并生成一个用于时间步推
                进的更新函数 update(u_0, u_1, phi_0, phi_1, dt)，在每个时间步调用时返回当前的离散
                矩阵 A 与右端项 L。
                使用示例：在已有相场空间 phispace 的基础上，传入物理参数，设置积分精度与稳定参数，连接
                该节点并调用其输出 update 函数即可实现时间步进求解。
                """
    INPUT_SLOTS = [
        PortConf("epsilon", DataType.FLOAT, title="界面厚度参数"),
        PortConf("Pe", DataType.FLOAT, title="Peclet 数"),
        PortConf("phispace", DataType.SPACE, title="相场函数空间"),
        PortConf("q", DataType.INT, 0, default = 5, min_val=3, title="积分精度"),
        PortConf("s", DataType.FLOAT, 0, title="稳定参数", default=1.0)
    ]
    OUTPUT_SLOTS = [
        PortConf("update", DataType.FUNCTION, title="更新函数")
    ]
    @staticmethod
    def run(epsilon, Pe, phispace, q, s):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.fem import (ScalarMassIntegrator, ScalarConvectionIntegrator,
                                SourceIntegrator,ScalarDiffusionIntegrator)
        from fealpy.fem import (BilinearForm, LinearForm, BlockForm, LinearBlockForm)
        
        def update(u_0, u_1, phi_0, phi_1, dt):
            cm = 1/Pe
            cf = 1.0
            ci = epsilon ** 2
            
            A00 = BilinearForm(phispace)
            BM_phi = ScalarMassIntegrator(q=q)
            BM_phi.coef = 3/(2*dt)
            
            BC_phi = ScalarConvectionIntegrator(q=q) 
            BC_phi.coef = 2*u_1

            A00.add_integrator(BM_phi)
            A00.add_integrator(BC_phi)

            A01 = BilinearForm(phispace)
            BD_phi = ScalarDiffusionIntegrator(q=q)
            BD_phi.coef = cm
            
            A01.add_integrator(BD_phi)
            
            A10 = BilinearForm(phispace)
            BD_mu = ScalarDiffusionIntegrator(q=q)
            BD_mu.coef = -ci

            BM_mu0 = ScalarMassIntegrator(q=q)
            BM_mu0.coef = -s * cf

            A10.add_integrator(BD_mu)
            A10.add_integrator(BM_mu0)

            A11 = BilinearForm(phispace)
            BM_mu1 = ScalarMassIntegrator(q=q)
            BM_mu1.coef = 1

            A11.add_integrator(BM_mu1)  

            A = BlockForm([[A00, A01], [A10, A11]]) 


            L0 = LinearForm(phispace)
            LS_phi = SourceIntegrator(q=q)
            @barycentric
            def LS_phi_coef(bcs, index):
                result = (4*phi_1(bcs, index) - phi_0(bcs, index))/(2*dt)
                result += bm.einsum('jid, jid->ji', u_0(bcs, index), phi_1.grad_value(bcs, index))
                return result

            LS_phi.source = LS_phi_coef
            L0.add_integrator(LS_phi)

            L1 = LinearForm(phispace)
            LS_mu = SourceIntegrator(q=q)
            @barycentric
            def LS_mu_coef(bcs, index): 
                result = -2*(1+s)*phi_1(bcs, index) + (1+s)*phi_0(bcs, index)
                result += 2*phi_1(bcs, index)**3 - phi_0(bcs, index)**3
                result *= cf
                return result
            LS_mu.source = LS_mu_coef

            L1.add_integrator(LS_mu)

            L = LinearBlockForm([L0, L1])
            return A, L
        return update




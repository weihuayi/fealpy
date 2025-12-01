
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNSNewton", "StationaryNSOssen", "StationaryNSStokes"]


class StationaryNSNewton(CNodeType):
    r"""Stationary incompressible Navier-Stokes solver using Newton iteration.

    Inputs:
        constitutive (int): Constitutive equation type (1 for standard viscous, 2 for general viscous work).
        mu (float): Dynamic viscosity coefficient.
        rho (float): Fluid density or a function returning density values.
        source (function): Source term function in the momentum equation.
        uspace (space): Velocity finite element function space.
        pspace (space): Pressure finite element function space.
        q (int, optional): Quadrature degree for numerical integration (default=3, minimum=3).

    Outputs:
        BForm (linops): Block bilinear form representing the linearized Navier-Stokes operator.
        LForm (linops): Block linear form representing the right-hand side vector.
        update (function): Function that updates the coefficients of the bilinear and linear forms 
                           based on the current velocity iterate.
    """
    TITLE: str = "稳态 NS 方程 Newton 迭代格式"
    PATH: str = "simulation.discretization"
    DESC: str = """该节点实现稳态不可压 Navier-Stokes 方程的 Newton 线性化格式，构建线性化的双线性与线性算子，
                并提供系数更新函数以迭代修正速度场，实现非线性方程的有限元求解。"""
    INPUT_SLOTS = [
        PortConf("constitutive", DataType.MENU, 0, title="本构方程", default=1, items=[i for i in range(1, 2)]),
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("q", DataType.INT, 0, default = 3, min_val=3, title="积分精度")
    ]
    OUTPUT_SLOTS = [
        PortConf("BForm", DataType.LINOPS, title="算子"),
        PortConf("LForm", DataType.LINOPS, title="向量"),
        PortConf("update", DataType.FUNCTION, title="更新函数")
    ]
    @staticmethod
    def run(constitutive, mu, rho, source, uspace, pspace, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
        from fealpy.fem import (ScalarMassIntegrator,ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                                SourceIntegrator, ViscousWorkIntegrator)
 
        A00 = BilinearForm(uspace)
        u_BM_netwon = ScalarMassIntegrator(q = q)
        u_BC = ScalarConvectionIntegrator(q = q)
        if constitutive == 1:
            u_BVW = ScalarDiffusionIntegrator(q=q)
        elif constitutive == 2:
            u_BVW = ViscousWorkIntegrator(q=q)
        
        A00.add_integrator(u_BM_netwon)
        A00.add_integrator(u_BC)
        A00.add_integrator(u_BVW)
        
        A01 = BilinearForm((pspace, uspace))
        u_BPW = PressWorkIntegrator(q = q)
        A01.add_integrator(u_BPW)
       
        A = BlockForm([[A00, A01], [A01.T, None]]) 

        L0 = LinearForm(uspace)
        u_LSI = SourceIntegrator(q = q)
        u_source_LSI = SourceIntegrator(q = q)
        L0.add_integrator(u_LSI) 
        L0.add_integrator(u_source_LSI)
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        
        def update(u0): 
            cv = mu
            cc = rho
            pc = 1.0
            cbf = source
            
            ## BilinearForm
            u_BVW.coef = cv
            u_BPW.coef = -pc

            @barycentric
            def u_BC_coef(bcs, index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                return cccoef * u0(bcs, index)
            u_BC.coef = u_BC_coef

            @barycentric
            def u_BM_netwon_coef(bcs,index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                return cccoef * u0.grad_value(bcs, index)
            u_BM_netwon.coef = u_BM_netwon_coef

            ## LinearForm 
            @barycentric
            def u_LSI_coef(bcs, index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result = cccoef*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
                return result
            u_LSI.source = u_LSI_coef
            u_source_LSI.source = cbf

        
        return (A, L, update)


class StationaryNSOssen(CNodeType):
    r"""Stationary incompressible Navier-Stokes solver using Oseen iteration.

    Inputs:
        constitutive (int): Constitutive equation type (1 for standard viscous, 2 for general viscous work).
        mu (float): Dynamic viscosity coefficient.
        rho (float or callable): Fluid density or a function returning density values.
        source (function): Source term function in the momentum equation.
        uspace (space): Velocity finite element function space.
        pspace (space): Pressure finite element function space.
        q (int, optional): Quadrature degree for numerical integration (default=3, minimum=3).

    Outputs:
        BForm (linops): Block bilinear form representing the Oseen linearized Navier-Stokes operator.
        LForm (linops): Block linear form representing the right-hand side vector.
        update (function): Function that updates the coefficients of the bilinear and linear forms 
                           based on the current velocity iterate.
    """
    TITLE: str = "稳态 NS 方程 Ossen 迭代格式"
    PATH: str = "simulation.discretization"
    DESC: str = """该节点实现稳态不可压 Navier-Stokes 方程的 Oseen 线性化格式，通过构建对流、扩散及压力耦合算子并提供
                系数更新函数，实现对非线性系统的线性近似与迭代求解。"""
    INPUT_SLOTS = [
        PortConf("constitutive", DataType.MENU, 0, title="本构方程", default=1, items=[i for i in range(1, 2)]),
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("q", DataType.INT, 0, default = 3, min_val=3, title="积分精度")
    ]
    OUTPUT_SLOTS = [
        PortConf("BForm", DataType.LINOPS, title="算子"),
        PortConf("LForm", DataType.LINOPS, title="向量"),
        PortConf("update", DataType.FUNCTION, title="更新函数")
    ]
    @staticmethod
    def run(constitutive, mu, rho, source, uspace, pspace, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
        from fealpy.fem import (ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                                SourceIntegrator, ViscousWorkIntegrator)
 
        A00 = BilinearForm(uspace)
        u_BC = ScalarConvectionIntegrator(q=q)
        if constitutive == 1:
            u_BVW = ScalarDiffusionIntegrator(q=q)
        elif constitutive == 2:
            u_BVW = ViscousWorkIntegrator(q=q)
        
        A00.add_integrator(u_BC)
        A00.add_integrator(u_BVW)

        A01 = BilinearForm((pspace, uspace))
        u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(u_BPW)
        
        A = BlockForm([[A00, A01], [A01.T, None]]) 

        L0 = LinearForm(uspace)
        u_source_LSI = SourceIntegrator(q=q)
        L0.add_integrator(u_source_LSI) 
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])

        def update(u0): 
            cv = mu
            cc = rho
            pc = 1.0
            cbf = source
            
            ## BilinearForm
            u_BVW.coef = cv
            u_BPW.coef = -pc

            @barycentric
            def u_BC_coef(bcs, index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                return cccoef * u0(bcs, index)
            u_BC.coef = u_BC_coef

            ## LinearForm 
            u_source_LSI.source = cbf

        return A, L, update

class StationaryNSStokes(CNodeType):
    r"""Finite element discretization of steady incompressible Navier-Stokes equations 
        using Stokes formulation.

    Inputs:
        constitutive (int): Constitutive equation type (1 for standard viscous, 2 for 
                            general viscous work).
        mu (float): Dynamic viscosity coefficient.
        rho (float): Fluid density or a function returning density values.
        source (function): Source term function in the momentum equation.
        uspace (space): Velocity finite element function space.
        pspace (space): Pressure finite element function space.
        q (int, optional): Quadrature degree for numerical integration (default=3, minimum=3).
        
    Outputs:
        BForm (linops): Block bilinear form representing the Stokes operator.
        LForm (linops): Block linear form representing the right-hand side vector.
        update (function): Function that updates the coefficients of the bilinear and 
                            linear forms.
    """
    TITLE: str = "稳态 NS 方程 Stokes 迭代格式"
    PATH: str = "simulation.discretization"
    DESC: str = """基于有限元构建稳态不可压 Stokes 型离散算子，组装黏性与压强耦合项并提供
                系数更新函数，用于稳态流场的有限元求解。"""
    INPUT_SLOTS = [
        PortConf("constitutive", DataType.MENU, 0, title="本构方程", default=1, items=[i for i in range(1, 2)]),
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),   
        PortConf("q", DataType.INT, 0, default = 3, min_val=3, title="积分精度")
    ]
    OUTPUT_SLOTS = [
        PortConf("BForm", DataType.LINOPS, title="算子"),
        PortConf("LForm", DataType.LINOPS, title="向量"),
        PortConf("update", DataType.FUNCTION, title="更新函数")
    ]
    @staticmethod
    def run(constitutive, mu, rho, source, uspace, pspace, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
        from fealpy.fem import (PressWorkIntegrator, ScalarDiffusionIntegrator,
                            SourceIntegrator, ViscousWorkIntegrator)
        from fealpy.decorator import barycentric

        A00 = BilinearForm(uspace)
        
        if constitutive == 1:
            u_BVW = ScalarDiffusionIntegrator(q=q)
        elif constitutive == 2:
            u_BVW = ViscousWorkIntegrator(q=q)
        
        A00.add_integrator(u_BVW)
        
        A01 = BilinearForm((pspace, uspace))
        u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(u_BPW)

        A = BlockForm([[A00, A01], [A01.T, None]]) 

        L0 = LinearForm(uspace)
        u_LSI = SourceIntegrator(q=q)
        u_source_LSI = SourceIntegrator(q=q)
        L0.add_integrator(u_LSI) 
        L0.add_integrator(u_source_LSI)
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])

        def update(u0): 
            cv = mu
            cc = rho
            pc = 1.0
            cbf = source
            
            ## BilinearForm
            u_BVW.coef = cv
            u_BPW.coef = -pc

            ## LinearForm 
            @barycentric
            def u_LSI_coef(bcs, index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result = -cccoef*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
                return result
            u_LSI.source = u_LSI_coef
            u_source_LSI.source = cbf

        return A, L, update
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['GaugeUzawaNSSimulation']

class GaugeUzawaNSSimulation(CNodeType):
    r"""Finite Element Discretization for the Gauge Uzawa Navier-Stokes Equation.

    This node constructs the finite element bilinear and linear forms
    required for the discretization of the Gauge Uzawa formulation of 
    the Navier-Stokes equations using the provided function spaces.
    
    Inputs:
        rho0 (float): Density of the fluid 0.
        rho1 (float): Density of the fluid 1.
        mu0 (float): Viscosity of the fluid 0.
        mu1 (float): Viscosity of the fluid 1.
        lam (float): stress coefficient
        gamma (float): Mobility parameter.
        uspace (SpaceType): Finite element space for the velocity field.
        pspace (SpaceType): Finite element space for the pressure field.
        phispace (SpaceType): Finite element space for the phase field φ.
        q (int): Quadrature degree used for numerical integration.
    """
    TITLE: str = "Gauge Uzawa Navier-Stokes 方程有限元离散"
    PATH: str = "simulation.discretization"
    DESC: str = """
                使用有限元方法对 Gauge Uzawa 形式的 Navier-Stokes 方程进行离散。
                本节点实现了速度场与压力场的有限元双线性形式与线性形式组装，适用于不可压缩流体动力学问题。
                通过输入流体密度、粘度、应力系数、函数空间 以及积分精度，自动构造相应的双线性形式与线性形式，
                并生成多个用于时间步推进的更新函数 
                update_us(phi_n , phi , u_n ,s_n , mv),
                update_ps(phi, us),
                update_velocity(phi, us, ps),
                update_gauge(us, s_n),
                在每个时间步调用时返回当前的离散矩阵 A 与右端项 b。
                """
    INPUT_SLOTS = [
        PortConf("rho0", DataType.FLOAT, title="第一液相密度"),
        PortConf("rho1", DataType.FLOAT, title="第二液相密度"),
        PortConf("mu0", DataType.FLOAT, title="第一液相粘度"),
        PortConf("mu1", DataType.FLOAT, title="第二液相粘度"),
        PortConf("lam", DataType.FLOAT, title="应力系数"),
        PortConf("gamma", DataType.FLOAT, title="迁移率参数"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("phispace", DataType.SPACE, title="相场函数空间"),
        PortConf("q", DataType.INT, 0, default = 4, min_val=3, title="积分精度"),
    ]
    OUTPUT_SLOTS = [
        PortConf("update_us", DataType.FUNCTION, title="辅助速度更新函数"),
        PortConf("update_ps", DataType.FUNCTION, title="伪压力更新函数"),
        PortConf("update_velocity", DataType.FUNCTION, title="速度更新函数"),
        PortConf("update_gauge", DataType.FUNCTION, title="规范变量更新函数"),
        PortConf("update_pressure", DataType.FUNCTION, title="压力更新函数")
    ]
    @staticmethod
    def run(rho0, rho1, mu0, mu1, lam, gamma,
            uspace, pspace,phispace, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.fem import (ScalarMassIntegrator, ScalarDiffusionIntegrator,
                                SourceIntegrator,ViscousWorkIntegrator, ScalarConvectionIntegrator)
        from fealpy.fem import (BilinearForm, LinearForm ,DirichletBC)
        mesh = uspace.mesh
        bar_mu = min(mu0, mu1)
        us_bform = BilinearForm(uspace)
        ps_bform = BilinearForm(pspace)
        u_bform = BilinearForm(uspace)
        s_bform = BilinearForm(pspace)
        p_bform = BilinearForm(pspace)
        
        us_lform = LinearForm(uspace)
        ps_lform = LinearForm(pspace)
        u_lform = LinearForm(uspace)
        s_lform = LinearForm(pspace)
        p_lform = LinearForm(pspace)
        
        us_SMI = ScalarMassIntegrator(q=q)
        us_SMI_phiphi = ScalarMassIntegrator(q=q)
        us_VWI = ViscousWorkIntegrator(q=q)
        us_SCI = ScalarConvectionIntegrator(q=q)
        us_SSI = SourceIntegrator(q=q)
        
        ps_SDI = ScalarDiffusionIntegrator(q=q)
        ps_SSI = SourceIntegrator(q=q)
        
        u_SMI = ScalarMassIntegrator(coef=1.0 , q=q)
        u_SSI = SourceIntegrator(q=q)
        
        s_SMI = ScalarMassIntegrator(coef=1.0 , q=q)
        s_SSI = SourceIntegrator(q=q)
        
        p_SMI = ScalarMassIntegrator(coef=1.0+1e-8 , q=q)
        p_SSI = SourceIntegrator(q=q)
        
        us_bform.add_integrator(us_SMI, us_VWI, us_SCI , us_SMI_phiphi)
        us_lform.add_integrator(us_SSI)
        ps_bform.add_integrator(ps_SDI)
        ps_lform.add_integrator(ps_SSI)
        u_bform.add_integrator(u_SMI)
        u_lform.add_integrator(u_SSI)
        s_bform.add_integrator(s_SMI)
        s_lform.add_integrator(s_SSI)
        p_bform.add_integrator(p_SMI)
        p_lform.add_integrator(p_SSI)
        
        bc = DirichletBC(uspace)
        
        def density(phi):
            tag0 = phi[:] >1
            tag1 = phi[:] < -1
            phi[tag0] = 1
            phi[tag1] = -1
            rho = phispace.function()
            rho[:] = 0.5 * (rho0 + rho1) + 0.5 * (rho0 - rho1) * phi
            return rho
        
        def viscosity(phi):
            tag0 = phi[:] >1
            tag1 = phi[:] < -1
            phi[tag0] = 1
            phi[tag1] = -1
            mu = phispace.function()
            mu[:] = 0.5 * (mu0 + mu1) + 0.5 * (mu0 - mu1) * phi
            return mu
        
        def update_us(phi_n , phi , u_n ,s_n ,dt,
                      velocity_force,velocity_dirichlet_bc, mv = None):
            # 移动网格行为产生的速度场 mv 可选
            if mv is None:
                def mv(bcs):
                    NC = mesh.number_of_cells()
                    GD = mesh.geo_dimension()
                    shape = (NC, bcs.shape[0], GD)
                    return bm.zeros(shape, dtype=bm.float64)

            mu = viscosity(phi)
            rho = density(phi)
            rho_n = density(phi_n)

            @barycentric
            def mass_coef(bcs,index):
                uh0_val = u_n(bcs, index)
                guh0_val = u_n.grad_value(bcs, index)
                rho1_val = rho(bcs, index)
                grho1_val = rho.grad_value(bcs, index)
                div_u_rho = bm.einsum('cqii,cq->cq', guh0_val, rho1_val) 
                div_u_rho += bm.einsum('cqd,cqd->cq', grho1_val, uh0_val) 
                result0 = rho1_val + 0.5 * dt * div_u_rho
                
                # 添加移动网格项
                result1 = 0.5*dt*bm.einsum('cqi,cqi->cq', grho1_val, mv(bcs, index))
                result = result0 - result1
                return result
            
            @barycentric
            def phiphi_mass_coef(bcs, index):
                gphi_n = phi_n.grad_value(bcs, index)
                gphi_gphi = bm.einsum('cqi,cqj->cqij', gphi_n, gphi_n)
                result = (lam * dt/gamma) * gphi_gphi
                return result
            
            @barycentric
            def convection_coef(bcs, index):
                uh_n_val = u_n(bcs, index)
                rho1_val = rho(bcs, index)
                result = rho1_val[...,None] * (uh_n_val - mv(bcs, index))
                return dt * result
            
            @barycentric
            def mon_source(bcs, index):
                gphi_n = phi_n.grad_value(bcs, index)
                result0 = bm.sqrt(rho_n(bcs, index)[..., None]) * bm.sqrt(rho(bcs, index)[..., None]) * u_n(bcs, index)
                result1 = dt * bar_mu * s_n.grad_value(bcs, index)
                result2 = lam/gamma * (phi(bcs, index) - phi_n(bcs, index))[..., None] * gphi_n
                mv_gardphi = bm.einsum('cqi,cqi->cq', gphi_n, mv(bcs, index))
                result3 = dt*lam/gamma * mv_gardphi[..., None] * gphi_n 
                result = result0 - result1 - result2 + result3
                return result
            
            @barycentric
            def force_source(bcs, index):
                ps = mesh.bc_to_point(bcs, index)
                result = dt * rho(bcs,index)[...,None] * velocity_force(ps)
                return result
            
            @barycentric
            def source(bcs, index):
                result0 = mon_source(bcs, index)
                result1 = force_source(bcs, index)
                result = result0 + result1
                return result
            
            us_SMI.coef = mass_coef
            us_SMI_phiphi.coef = phiphi_mass_coef
            us_VWI.coef = dt * mu
            us_SCI.coef = convection_coef
            us_SSI.source = source
                        
            A = us_bform.assembly()
            b = us_lform.assembly()
            
            bc.gd = velocity_dirichlet_bc
            A , b = bc.apply(A, b)
            return A , b
        
        def update_ps(phi, us):
            rho = density(phi)
            
            @barycentric
            def diffusion_coef(bcs, index):
                result = 1 / rho(bcs, index)
                return result
            
            @barycentric
            def source_coef(bcs, index):
                uh_grad_val = us.grad_value(bcs, index)
                div_u = bm.einsum('cqii->cq', uh_grad_val)
                return div_u
            
            ps_SDI.coef = diffusion_coef
            ps_SSI.source = source_coef
            
            A = ps_bform.assembly()
            b = ps_lform.assembly()
            return A , b
        
        def update_velocity(phi, us, ps):
            rho = density(phi)
            
            @barycentric
            def source_coef(bcs, index):
                result = us(bcs, index)
                result += (1/rho(bcs, index)[..., None] )* ps.grad_value(bcs, index)
                return result
            
            u_SSI.source = source_coef
            
            A = u_bform.assembly()
            b = u_lform.assembly()
            return A , b
        
        def update_gauge(s_n, us):
            @barycentric
            def source_coef(bcs,index):
                result = s_n(bcs,index) - bm.einsum('cqii->cq', us.grad_value(bcs,index))
                return result
            s_SSI.source = source_coef

            A = s_bform.assembly()
            b = s_lform.assembly()
            return A , b
        
        def update_pressure(s, ps, dt):
            @barycentric
            def source_coef(bcs, index):
                result = -1/dt * ps(bcs, index)
                result +=  bar_mu * s(bcs, index)
                return result
            p_SSI.source = source_coef
            
            A = p_bform.assembly()
            b = p_lform.assembly()
            return A , b
        
        return update_us, update_ps, update_velocity, update_gauge, update_pressure
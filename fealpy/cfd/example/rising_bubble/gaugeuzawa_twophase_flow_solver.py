from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import DirichletBC,BlockForm
from fealpy.fem import ScalarMassIntegrator, ScalarDiffusionIntegrator, ScalarConvectionIntegrator
from fealpy.fem import SourceIntegrator, ViscousWorkIntegrator,ScalarSourceIntegrator    

from fealpy.decorator import barycentric,cartesian
from fealpy.solver import spsolve
from fealpy.sparse import COOTensor


class GaugeUzawaTwoPhaseFlowSolver:
    """
    A Gauge-Uzawa method based two-phase flow solver for coupled phase-field and Navier-Stokes equations.
    
    This solver uses operator splitting method to decompose the two-phase flow problem into phase-field 
    and fluid momentum equation subproblems, handling velocity-pressure coupling and phase-field-flow 
    coupling through time stepping and iterative solving.

    Parameters:
        pde(TwoPhaseFlowPDE): PDE problem class containing two-phase flow definitions including 
            phase-field equation, momentum equation, and physical parameters.
        
        mesh(Mesh): Computational mesh for discretizing the solution domain.
        
        up(int): Polynomial degree for velocity space, default is 2.
        
        pp(int): Polynomial degree for pressure space, default is 1.
        
        phip(int): Polynomial degree for phase-field space, default is 2.
        
        dt(float): Time step size, default is 0.01.
        
        q(int): Quadrature order for numerical integration, default is 4.

    Attributes:
        mesh(Mesh): Computational mesh.
        
        uspace(TensorFunctionSpace): Velocity function space (vector space).
        
        pspace(LagrangeFESpace): Pressure function space (scalar space).
        
        phispace(LagrangeFESpace): Phase-field function space (scalar space).
        
        uh(Function): Current time velocity field.
        
        ph(Function): Current time pressure field.
        
        phi(Function): Current time phase-field.

    Methods:
        assemble_phase_field_matrix(): Assemble phase-field equation coefficient matrix.
        
        assemble_momentum_matrix(): Assemble momentum equation coefficient matrix.
        
        solve_phase_field(): Solve phase-field equation subproblem.
        
        solve_momentum(): Solve momentum equation subproblem.
        
        time_step(): Execute one complete time step solution.
    """
    def __init__(self, pde, mesh, up=2, pp=1, phip=2, dt=0.01, q=4 ,
                 method=None,solver='direct',is_adaptive = False):
        self.pde = pde
        self.mesh = mesh
        self.dt = dt
        self.q = q
        self.time = 0.0
        self.up = up  # Velocity polynomial degree
        self.pp = pp  # Pressure polynomial degree
        self.phip = phip  # Phase-field polynomial degree
        self.method = method  # Assembly method
        self.solver = solver  # Solver method
        self.is_adaptive = is_adaptive
        
        # Create quadrature formula
        qf = mesh.quadrature_formula(q)
        self.bcs,self.ws = qf.get_quadrature_points_and_weights()
        # Create function spaces
        usspace = LagrangeFESpace(mesh, p=up)  # Velocity scalar space
        self.uspace = TensorFunctionSpace(usspace, (mesh.GD, -1))  # Velocity vector space
        self.pspace = LagrangeFESpace(mesh, p=pp)  # Pressure space
        self.phispace = LagrangeFESpace(mesh, p=phip)  # Phase-field space

        self.variable_initialize()

        self.set_initial_condition()  # Set initial conditions
        self.set_linear_system()
        self.set_lagrange_multiplier()
        
    def variable_initialize(self):
        """
        Initialize variables for a new simulation.
        """
        # Initialize functions
        self.uh = self.uspace.function()  # Current velocity
        self.uh0 = self.uspace.function()  # Previous time velocity
        self.ph = self.pspace.function()  # Current pressure
        self.phi = self.phispace.function()  # Current phase-field
        self.phi0 = self.phispace.function()  # Previous time phase-field
        
        self.mu = self.phispace.function()  # Viscosity field
        self.rho0 = self.phispace.function()  # Density field
        self.rho = self.phispace.function()  # Updated Density field
        self.f = self.phispace.function()  # 
        
        # Intermediate variables
        self.us = self.uspace.function()  # Intermediate velocity
        self.s = self.pspace.function()  # guage variable
        self.ps = self.pspace.function()  # Intermediate pressure
        self.rho = self.phispace.function()  # Density field
        
        self.bar_mu = bm.min(bm.array([self.pde.mu1, self.pde.mu2]))
        
    def set_linear_system(self):
        phispace = self.phispace
        uspace = self.uspace
        pspace = self.pspace
        
        self.phi_bform = BilinearForm(phispace)
        self.u_bform = BilinearForm(uspace)
        self.p_bform = BilinearForm(pspace)
        self.u_assembly_bform = BilinearForm(uspace)
         
        self.u_assembly_lform = LinearForm(uspace)
        self.phi_lform = LinearForm(phispace)
        self.u_lform = LinearForm(uspace)
        self.p_lform = LinearForm(pspace)
        
        gamma = self.pde.gamma
        eta = self.pde.eta
        dt = self.dt
        q = self.q
        method = self.method
        
        self.phi_SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q,method=method)
        self.phi_SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        self.phi_SCI = ScalarConvectionIntegrator(q=q , method=method)
        self.phi_SSI = ScalarSourceIntegrator(q = q,method=method)
        
        self.u_SMI = ScalarMassIntegrator(q=q, method=method)
        self.u_SMI_phiphi = ScalarMassIntegrator(q=q,method=method)
        self.u_VWI = ViscousWorkIntegrator(q=q)
        self.u_SCI = ScalarConvectionIntegrator(q=q, method=method)
        self.u_SSI = ScalarSourceIntegrator(q=q, method=method)
        
        self.p_SDI = ScalarDiffusionIntegrator(q=q, method=method)
        # self.p_SMI = ScalarMassIntegrator(coef=1e-10,q=q, method=method)
        self.p_SSI = ScalarSourceIntegrator(q=q, method=method)
        
        self.u_assembly_SMI = ScalarMassIntegrator(coef=1.0, q=q, method=method)
        self.u_assembly_SSI = ScalarSourceIntegrator(q=q, method=method)
        
        self.phi_bform.add_integrator(self.phi_SMI,self.phi_SDI,self.phi_SCI)
        self.phi_lform.add_integrator(self.phi_SSI)
        self.u_bform.add_integrator(self.u_SMI,self.u_SMI_phiphi, self.u_VWI, self.u_SCI)
        self.u_lform.add_integrator(self.u_SSI)
        
        self.p_bform.add_integrator(self.p_SDI)
        self.p_lform.add_integrator(self.p_SSI)
        
        self.u_assembly_bform.add_integrator(self.u_assembly_SMI)
        self.u_assembly_lform.add_integrator(self.u_assembly_SSI)
           
        self.u_bc = DirichletBC(uspace)
    
    def set_initial_condition(self):
        """
        Set initial conditions by interpolating analytical solutions to finite element functions.

        Parameters:
            t(float): Initial time, default is 0.0.

        Returns:
            None: This method modifies class attributes directly, no return value.
        """
        # Set initial velocity
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.init_velocity(p))
        self.uh[:] = self.uh0[:]
        
        # Set initial pressure
        self.ph[:] = self.pspace.interpolate(lambda p: self.pde.init_pressure(p))
        
        # Set initial phase-field
        self.phi0[:] = self.phispace.interpolate(lambda p: self.pde.init_phase(p))
        self.phi[:] = self.phi0[:]
        self.rho[:] = self.density(self.phi)
    
    def density(self, phi=None):
        """
        Update density field according to ρ = 0.5(ρ₁ - ρ₂)φ + 0.5(ρ₁ + ρ₂), where φ is the phase-field variable.

        Parameters:
            phi(Function): Phase-field function, default is None.

        Returns:
            Function: Density function computed from phase-field.
        """
        if phi is None:
            phi = self.phi
        tag0 = phi[:] >1
        tag1 = phi[:] < -1
        phi[tag0] = 1
        phi[tag1] = -1
        
        rho_1 = self.pde.rho_1
        rho_2 = self.pde.rho_2

        rho = 0.5 * (rho_1 - rho_2) * phi[:] + 0.5 * (rho_1 + rho_2)
        return rho  
    
    def viscosity(self, phi=None):
        """
        Update viscosity field according to μ = μ₁ + (μ₂ - μ₁)φ, where φ is the phase-field variable.

        Parameters:
            phi(Function): Phase-field function, default is None.

        Returns:
            Function: Viscosity function computed from phase-field.
        """
        if phi is None:
            phi = self.phi
        tag0 = phi[:] >1
        tag1 = phi[:] < -1
        phi[tag0] = 1
        phi[tag1] = -1
        
        mu_1 = self.pde.mu1
        mu_2 = self.pde.mu2

        mu = phi.space.function()
        mu[:] = 0.5 * (mu_1 - mu_2) * phi[:] + 0.5 * (mu_1 + mu_2)
        return mu

    def fphi(self, phi):
        """
        Calculate the phase-field force term f(φ) = (φ³ - φ)/η².

        Parameters:
            phi(Function): Phase-field function.

        Returns:
            Function: Phase-field force term evaluated at φ.
        """
        eta = self.pde.eta
        tag0 = phi[:] > 1
        tag1 = (phi[:] >= -1) & (phi[:] <= 1)
        tag2 = phi[:] < -1
        
        f = phi.space.function()
        f[tag0] = 2/eta**2  * (phi[tag0] - 1)
        f[tag1] = (phi[tag1]**3 - phi[tag1]) / (eta**2)
        f[tag2] = 2/eta**2 * (phi[tag2] + 1)
        f[:] = (phi[:]**3 - phi[:]) / (eta**2)
        return f
    
    def set_lagrange_multiplier(self):
        """
        Set the Lagrange multiplier for the weak formulation of the Allen-Cahn equation.
        This method directly introduces the Lagrange multiplier into the left-hand matrix for implicit solving.
        """
        
        LagLinearForm = LinearForm(self.phispace)
        Lag_SSI = ScalarSourceIntegrator(source=1, q=self.q , method=self.method)
        LagLinearForm.add_integrator(Lag_SSI)
        LagA = LagLinearForm.assembly()
        A0 = -self.dt * self.pde.gamma * bm.ones(self.phispace.number_of_global_dofs())
        
        self.A0 = COOTensor(bm.array([bm.arange(len(A0), dtype=bm.int32), 
                                 bm.zeros(len(A0), dtype=bm.int32)]), A0, spshape=(len(A0), 1))
        self.A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA,
                                spshape=(1, len(LagA)))
        self.b0 = bm.array([self.mesh.integral(self.pde.init_phase)], dtype=bm.float64)
        
    def lagrange_multiplier(self,A,b):
        """
        Extend the matrix and the right-hand side vector
        
        Parameters:
            A (CSRTensor): The original matrix from the bilinear form.
            b (TensorLike): The original right-hand side vector from the linear form.
        Returns:
            A (CSRTensor): The extended matrix with Lagrange multiplier terms.
            b (TensorLike): The extended right-hand side vector with Lagrange multiplier terms.
        """
        A0 = self.A0
        A1 = self.A1
        b0 = self.b0
        A = BlockForm([[A, A0], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b  = bm.concat([b, b0], axis=0)
        return A, b
    
    def lagrange_multi_exp_p(self,matrix,space:LagrangeFESpace):
        """
        对矩阵施加积分乘子的扩张.其添加的维度为最后一维.
        V = int_Omega  phi_i dx
        A* = [[A ,V^T],
              [V , 0 ]]
        Parameters:
            matrix: 需要扩张的矩阵
            space: 该矩阵对应的有限元空间
        Returns:
            matrix_exp: 扩张后的矩阵
        """
        gdofs = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        basis = space.basis(self.bcs)
        cm = self.mesh.entity_measure("cell")
        V_cell = bm.einsum('cqi , c , q -> ci', basis , cm , self.ws)
        V = bm.zeros((gdofs,), dtype=bm.float64)
        V = bm.index_add(V , (cell2dof,) , V_cell)
        Vn = bm.sqrt(bm.sum(V*V))
        if Vn != 0:
           V = V / Vn
        A0 = COOTensor(bm.array([bm.arange(len(V), dtype=bm.int32), 
                                 bm.zeros(len(V), dtype=bm.int32)]), V, spshape=(len(V), 1))
        
        blockform = BlockForm([ [matrix , A0 ] ,
                            [A0.T   , None] ])
        A_star = blockform.assembly_sparse_matrix(format='csr')
        return A_star
    
    def phase_field_assembly(self, t1=None, phi0=None, uh0=None, mv=None):
        dt = self.dt
        gamma = self.pde.gamma
        eta = self.pde.eta
        bcs = self.bcs
        if phi0 is None:
            phi0 = self.phi0            
        if uh0 is None:
            uh0 = self.uh0
        if t1 is None:
            t1 = self.time

        self.phi_SCI.coef = dt * (uh0(bcs) - mv(bcs))
        phi_force = self.phispace.interpolate(lambda p: self.pde.phi_force(p, t1))

        @barycentric
        def source_coef(bcs, index):
            phi_val = phi0(bcs, index)  
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * self.fphi(phi0)(bcs, index)
            result += dt * phi_force(bcs, index)
            return result
        
        self.phi_SSI.source = source_coef
        phi_A = self.phi_bform.assembly()
        phi_b = self.phi_lform.assembly()
        phi_A, phi_b = self.lagrange_multiplier(phi_A, phi_b)
        
        sphi = spsolve(phi_A, phi_b, solver='scipy')
        new_phi = sphi[:-1]
        return new_phi
        
    def momentum_assembly(self, t1=None, phi0=None, uh0=None, phi=None, s=None,mv=None):
        dt = self.dt
        lam = self.pde.lam
        gamma = self.pde.gamma
        
        if phi0 is None:
            phi0 = self.phi0            
        if uh0 is None:
            uh0 = self.uh0        
        if phi is None:
            phi = self.phi
        if s is None:
            s = self.s
            
        self.mu[:] = self.viscosity(phi)
        self.rho0[:] = self.density(phi0)
        self.rho[:] = self.density(phi)
        
        # Update coefficients
        @barycentric
        def mass_coef(bcs,index):
            uh0_val = uh0(bcs, index)
            guh0_val = uh0.grad_value(bcs, index)
            rho1_val = self.rho(bcs, index)
            grho1_val = self.rho.grad_value(bcs, index)
            div_u_rho = bm.einsum('cqii,cq->cq', guh0_val, rho1_val) 
            div_u_rho += bm.einsum('cqd,cqd->cq', grho1_val, uh0_val) 
            result = rho1_val + 0.5 * dt * div_u_rho
            # 添加移动网格项
            result -= 0.5*dt*bm.einsum('cqi,cqi->cq', grho1_val, mv(bcs, index))
            return result
        
        @barycentric
        def phiphi_mass_coef(bcs, index):
            gphi0 = phi0.grad_value(bcs, index)
            gphi_gphi = bm.einsum('cqi,cqj->cqij', gphi0, gphi0)
            result = (lam * dt / gamma) * gphi_gphi
            return result
        
        @barycentric
        def conv_coef(bcs, index):
            rhou = self.rho(bcs, index)[..., None] * (uh0(bcs, index) - mv(bcs, index))
            return dt * rhou
        
        @barycentric
        def momentum_source(bcs, index):
            gphi0_val = phi0.grad_value(bcs, index)
            result0 = bm.sqrt(self.rho0(bcs, index)[..., None]) * bm.sqrt(self.rho(bcs, index)[..., None]) * uh0(bcs, index)
            result1 = dt * self.bar_mu * s.grad_value(bcs, index)
            result2 = lam/gamma * (phi(bcs, index) - phi0(bcs, index))[..., None] * gphi0_val
            #  + dt*lam*self.theta * gphi0_val
            mv_gardphi = bm.einsum('cqi,cqi->cq', gphi0_val, mv(bcs, index))
            result3 = dt*lam/gamma * mv_gardphi[..., None] * gphi0_val 
            result = result0 - result1 - result2 + result3 
            return result
        
        @barycentric
        def f_source(bcs, index):
            val = uh0(bcs, index)
            val[...,0] = 0
            val[...,1] = -1
            val = self.uspace.interpolate(lambda p: self.pde.mom_force(p, t1))(bcs, index)
            # result = dt*self.rho(bcs, index)[..., None] * val
            result = dt*val
            return result

        @barycentric
        def mom_source(bcs, index):
            result0 = momentum_source(bcs, index)
            result1 = f_source(bcs, index)
            result = result0 + result1
            return result
        
        self.u_SMI.coef = mass_coef
        self.u_SMI_phiphi.coef = phiphi_mass_coef
        self.u_VWI.coef =  dt * self.mu
        self.u_SCI.coef = conv_coef
        self.u_SSI.source = mom_source
        
        u_A = self.u_bform.assembly()
        u_b = self.u_lform.assembly()

        u_A,u_b = self.u_bc.apply(u_A, u_b)
        u_new = spsolve(u_A, u_b, solver='scipy')
        return u_new
    
    def pressure_correction_assembly(self, phi=None, us=None):
        """
        Assemble the pressure correction equation coefficient matrix and right-hand side vector.
        
        The pressure correction equation is given by:
        ∇·u = 0, where u is the velocity field.
        
        Returns:
            tuple: Tuple containing coefficient matrix A and right-hand side vector b (A, b).
        """
        if phi is None:
            phi = self.phi
        if us is None:
            us = self.us

        @barycentric
        def diff_coef(bcs, index):
            result = self.rho(bcs, index)
            return 1/result
        self.p_SDI.coef = diff_coef
        @barycentric
        def div_source(bcs, index):
            u_grad = us.grad_value(bcs, index)
            div_u = bm.einsum('cqii->cq', u_grad)
            return div_u
        self.p_SSI.source = div_source
        
        p_A = self.p_bform.assembly()
        p_b = self.p_lform.assembly()
        p_correction = spsolve(p_A, p_b, solver='scipy')
        return p_correction
    
    def update_velocity(self, phi=None, us=None, ps=None):
        if phi is None:
            phi = self.phi
        if us is None:
            us = self.us
        if ps is None:
            ps = self.ps

        # update next time velocity
        bform = self.u_assembly_bform
        A = bform.assembly()

        lform = self.u_assembly_lform
        @barycentric
        def source_coef(bcs, index):
            result = us(bcs, index)
            result += (1/self.rho(bcs, index)[..., None] )* ps.grad_value(bcs, index)
            return result

        self.u_assembly_SSI.source = source_coef
        b = lform.assembly()
        u_new = spsolve(A, b, solver='scipy') 
        return u_new

    def update_gauge(self, s0=None, us=None):
        if s0 is None:
            s0 = self.s
        if us is None:
            us = self.us
        @barycentric
        def source_coef(bcs,index):
            result = s0(bcs,index) - bm.einsum('cqii->cq', us.grad_value(bcs,index))
            return result
        # update next time guage variable
        bform = BilinearForm(self.pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=1.0, q=self.q))
        A = bform.assembly()

        lform = LinearForm(self.pspace)
        lform.add_integrator(SourceIntegrator(source=source_coef, q=self.q))
        b = lform.assembly()

        s_new = spsolve(A, b, solver='scipy')
        return s_new

    def update_pressure(self, s1=None, ps1=None):
        if s1 is None:
            s1 = self.s
        if ps1 is None:
            ps1 = self.ps
        # update next time pressure
        bform = BilinearForm(self.pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=1.0, q=self.q))
        A = bform.assembly()
        # A = self.lagrange_multi_exp_p(A,self.pspace)
        lform = LinearForm(self.pspace)
        @barycentric
        def source_coef(bcs, index):
            result = -1/self.dt * ps1(bcs, index)
            result +=  self.pde.mu * s1(bcs, index)
            return result
        
        lform.add_integrator(SourceIntegrator(source=source_coef, q=self.q))
        b = lform.assembly()
        # b = bm.concat([b, bm.array([0.0], dtype=bm.float64)], axis=0)
        print("Solving for pressure with Lagrange multiplier...")
        p_new = spsolve(A, b, solver='scipy')
        return p_new
    
    def time_step(self,t ,phi0 , uh0 , mv = None):
        """
        Perform a single time step of the simulation.
        """
        print(f"Time: {t:.4f}")
        if mv is None and t == self.dt:
            self.mspace = TensorFunctionSpace(self.phispace, (self.mesh.GD, -1))
            self.mv = self.mspace.function()
        else:
            self.mv = mv
        v_bc_func = lambda p: self.pde.velocity_dirichlet_bc(p, t)
        self.u_bc.gd = v_bc_func
        # Phase-field assembly
        self.phi[:] = self.phase_field_assembly(t1=t, phi0=phi0, uh0=uh0, mv=self.mv)
        
        # Momentum assembly
        self.us[:] = self.momentum_assembly(t1=t, phi0=phi0, uh0=uh0, phi=self.phi, s=self.s, mv=self.mv)
        
        # Pressure correction assembly
        self.ps[:] = self.pressure_correction_assembly(phi=self.phi, us=self.us)
        
        # Update velocity
        self.uh[:] = self.update_velocity(phi=self.phi, us=self.us, ps=self.ps)
        
        # Update gauge variable
        self.s[:] = self.update_gauge(s0=self.s, us=self.us)

        self.phi0[:] = self.phi[:]
        self.uh0[:] = self.uh[:]
        
        # self.ph[:] = self.update_pressure(s1=self.s, ps1=self.ps)

        return self.phi, self.ph, self.uh

    def save_vtu(self, step: int):
        """
        Save the current state of the simulation to a .vtu file.

        Parameters:
            step (int): The current time step index.
        """
        self.mesh.nodedata['interface'] = self.phi
        self.mesh.nodedata['velocity'] = self.uh.reshape(self.mesh.GD,-1).T
        self.mesh.nodedata['pressure'] = self.ph
        fname = './' + 'rising_bubble_50' + str(step).zfill(10) + '.vtu'
        self.mesh.to_vtk(fname=fname)
    
    def current_error(self,phi , p , u , t):
        """
        计算当前时间步的误差.
        
        Parameters:
            phi: 相场变量 phi
            p: 压力变量 p
            u: 速度变量 u
            t: 当前时间
        Returns:
            err_phi: 相场变量 phi 的误差
            err_p: 压力变量 p 的误差
            err_u: 速度变量 u 的误差
        """
        pde = self.pde
        mesh = self.mesh
        q = self.up + 2

        phi_exact_func = lambda p: pde.phase_solution(p , t)
        p_exact_func = lambda p: pde.pressure_solution(p , t)
        u_exact_func = lambda p: pde.velocity_solution(p , t)
        # phi_grad_exact_func = lambda p: pde.gradient_phase(p , t)
        # u_grad_exact_func = lambda p: pde.gradient_velocity(p , t)

        L2_err_phi = mesh.error(phi_exact_func , phi , q = q)
        L2_err_p = mesh.error(p_exact_func , p , q = q) 
        L2_err_u = mesh.error(u_exact_func , u , q = q)
        
        # H1_err_phi = mesh.error(phi_grad_exact_func , phi.grad_value , q = q)
        # H1_err_u = mesh.error(u_grad_exact_func , u.grad_value , q = q)

        return L2_err_phi, L2_err_p, L2_err_u
    
    def compute_bubble_centroid(self):
        """
        Compute the centroid of the bubble region (phi > 0).

        Parameters:
            mesh (TriangleMesh): The computational mesh.
            phi (Function): The phase field function.

        Returns:
            numpy.ndarray: The centroid position [x, y].
        """
        # 获取网格节点和相场值
        nodes = self.mesh.node  # 网格节点坐标
        cell = self.mesh.cell
        cell_to_dof = self.phispace.cell_to_dof()
        phi_cell_values = bm.mean(self.phi[cell_to_dof],axis=-1)  # 相场值

        # 筛选气泡区域（phi > 0）
        bubble_mask = phi_cell_values > 0
        bubble_bc_nodes = bm.mean(nodes[cell[bubble_mask]],axis=1)
        bubble_phi = phi_cell_values[bubble_mask]

        # 计算质心位置
        centroid = bm.sum(bubble_bc_nodes.T * bubble_phi, axis=1) / bm.sum(bubble_phi)
        return centroid
    
    def mass_compute(self):
        """
        Compute the mass of the bubble region (phi > 0).

        Returns:
            float: The total mass of the bubble region.
        """
        phi = self.phi
        d = self.d
        ws = self.ws
        bcs = self.bcs
        mass = bm.einsum('cq,cq,q->', phi(bcs), d, ws)
        return mass
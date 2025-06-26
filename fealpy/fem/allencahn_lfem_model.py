from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEDataManager, ComputationalModel
from ..model.phasefield.ac_circle_data_2d import AcCircleData2D
from ..decorator import variantmethod,barycentric

# FEM imports
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import DirichletBC
from ..fem import (ScalarDiffusionIntegrator,
                   ScalarConvectionIntegrator,
                   ScalarMassIntegrator,
                   ScalarSourceIntegrator)
from ..solver import spsolve,cg

class AllenCahnLFEMModel(ComputationalModel):
    """
    Allen-Cahn phase field model using Lagrange finite element method (LFEM).
    
    This model implements the Allen-Cahn equation in a weak form suitable for finite element analysis.
    It uses Lagrange finite element spaces for spatial discretization and supports time-stepping methods.
    """
    def __init__(self):
        super().__init__(pbar_log=True, log_level="INFO")
        # self.pdm = PDEDataManager("ac_circle")
    
    def set_pde(self,pde: Union[str, AcCircleData2D] = 'ac_circle'):
        """
        Set the PDE data for the model.
        
        Parameters:
            pde (str or AcCircleData2D): The name of the PDE or an instance of AcCircleData2D.
        """
        # if isinstance(pde, str):
        #     self.pde = self.pdm.get_example(pde)
        # elif isinstance(pde, AcCircleData2D):
        self.pde = pde
        # else:
        #     raise TypeError("pde must be a string or an instance of AcCircleData2D.")
        
    def set_init_mesh(self,nx:int = 10,ny:int = 10,meshtype: str = 'tri', **kwargs):
        """
        Set the initial mesh for the model.
        Parameters:
            meshtype (str): The type of mesh to create ('tri' for triangular mesh).
            **kwargs: Additional keyword arguments for mesh creation.
        """
        if meshtype == 'tri':
            from ..mesh import TriangleMesh
            domain = self.pde.domain()
            self.mesh = TriangleMesh.from_box(domain,nx,ny, **kwargs)
        elif meshtype == 'quad':
            from ..mesh import QuadrangleMesh
            self.mesh = QuadrangleMesh.from_box(self.pde.domain(),nx,ny, **kwargs)
        kwargs = bm.context(self.mesh.node)
        vertices = bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], **kwargs)
        self.mesh.nodedata["vertices"] = vertices
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int = 1):
        """
        Set the order of the Lagrange finite element space.
        
        Parameters:
            order (int): The polynomial order of the finite element space.
        """
        self.p = p
        self.logger.info(f"Finite element space set with polynomial order {self.p}.")
    
    def set_quadrature(self, q: int = 4):
        """
        Set the quadrature order for numerical integration.
        
        Parameters:
            q (int): The order of the quadrature rule.
        """
        self.q = q
        qf = self.mesh.quadrature_formula(q)
        self.bcs,self.ws = qf.get_quadrature_points_and_weights()
        self.logger.info(f"Quadrature order set to {self.q}.")

    def set_assemble_method(self, method: Union[str,None] = None):
        """
        Set the method for assembling the matrix.
        
        Parameters:
            method (str): The assembly method to use ('sparse' for sparse matrix assembly).
        """
        self.assemble_method = method
        self.logger.info(f"Assembly method set to {self.assemble_method}.")

    def set_function_space(self):
        """
        Set the function space for the model.
        
        This method initializes the Lagrange finite element space based on the mesh and polynomial order.
        """
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.logger.info(f"Function space initialized with {self.space.number_of_global_dofs()} degrees of freedom.")
        self.phi = self.space.function()
        self.phi0 = self.space.function()
        self.theta = 0.0

    def set_vector_function_space(self,up: int = 2):
        """
        Set the vector function space for the model.
        
        Parameters:
            up (int): The polynomial order for the vector function space.
        """
        from ..functionspace import TensorFunctionSpace
        usspace = LagrangeFESpace(self.mesh, p=up)
        self.uspace = TensorFunctionSpace(usspace, (self.mesh.GD,-1))
        self.uh0 = self.uspace.interpolate(lambda p : self.pde.velocity_field(p,t=0.0))
        self.logger.info(f"velocity function space initialized with polynomial order {up}.")

    def set_initial_condition(self,t = 0.0):
        """
        Set the initial condition for the phase field variable.
        This method initializes the phase field variable using the initial solution from the PDE data.

        Parameters:
            t (float): The initial time for the phase field variable.
        """
        pde = self.pde
        space = self.space
        self.t = t
        self.phi0[:] = space.interpolate(lambda p:pde.init_solution(p, t=t))
    
    def set_time_step(self, dt: float = 0.01):
        """
        Set the time step for the simulation.
        
        Parameters:
            dt (float): The time step size.
        """
        self.dt = dt
        self.logger.info(f"Time step set to {dt}.")

    def set_move_mesher(self, mmesher:str = 'GFMMPDE',
                        beta :float = 0.5,
                        tau :float = 0.5,
                        tmax :float = 0.5,
                        alpha :float = 0.5,
                        moltimes :int = 5,
                        monitor: str = 'arc_length',
                        mol_meth :str = 'projector',
                        config: Optional[dict] = None):
        """
        Set the mesher for moving the mesh during the simulation.

        Parameters:
            mm (str): The type of mesh mover to use (default is 'GFMMPDE').
            beta (float): The beta parameter for the monitor funtion (default is 0.5).
            tmax (float): The maximum time for the mesh mover (default is 0.
            alpha (float): The alpha parameter for the mesh mover (default is 0.5).
            moltimes (int): The number of times to move the mesh (default is 5).
            config (dict, optional): Additional configuration parameters for the mesh mover.
        """
        from ..functionspace import TensorFunctionSpace
        from ..mmesh.mmesher import MMesher
        self.mm = MMesher(self.mesh, 
                     uh = self.phi0 ,
                     space= self.space,
                     beta=beta,
                     ) 
        self.mm.config.active_method = mmesher
        self.mm.config.tau = tau
        self.mm.config.t_max = tmax
        self.mm.config.alpha = alpha
        self.mm.config.mol_times = moltimes
        self.mm.config.monitor = monitor
        self.mm.config.mol_meth = mol_meth
        self.mm.config.is_pre = False
        if config is not None:
            for key, value in config.items():
                try:
                    # check if the key exists in the config
                    getattr(self.mm.config, key)
                    # if it exists, set the value
                    setattr(self.mm.config, key, value)
                    self.logger.info(f"Set config.{key} = {value}")
                except AttributeError:
                    self.logger.warning(f"Config attribute '{key}' does not exist, skipping.")
        
        self.mm.initialize()
        self.node0 = self.mesh.node.copy()
        smspace = self.mm.instance.mspace
        self.mspace = TensorFunctionSpace(smspace, (self.mesh.GD,-1))
    
    @variantmethod('forward')
    def set_bilinear_form(self):
        """
        Set the bilinear form for the weak formulation of the Allen-Cahn equation.
        
        This method defines the bilinear form using diffusion and mass integrators.
        """
        space = self.space
        gamma = self.pde.gamma
        eta = self.pde.eta
        dt = self.dt
        q = self.q
        method = self.assemble_method
        self.bform = BilinearForm(space)
        self.SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q, method=method)
        self.SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        self.bform.add_integrator(self.SMI,self.SDI)
        
        self.logger.info("Bilinear form forward set with diffusion, and mass terms.")

    @set_bilinear_form.register('backward')
    def set_bilinear_form(self):
        """
        Set the bilinear form for the weak formulation of the Allen-Cahn equation.
        
        This method defines the bilinear form using diffusion, convection, and mass integrators.
        """
        space = self.space
        gamma = self.pde.gamma
        eta = self.pde.eta
        dt = self.dt
        q = self.q
        method = self.assemble_method
        self.bform = BilinearForm(space)
        self.SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q,method=method)
        self.SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        self.SCI = ScalarConvectionIntegrator(q=q, method=method)
        self.bform.add_integrator(self.SMI,self.SDI, self.SCI)
        
        self.logger.info("Bilinear form backward set with diffusion,convection and mass terms.")

    def set_linear_form(self):
        """
        Set the linear form for the weak formulation of the Allen-Cahn equation.
        
        This method defines the linear form using source integrator.
        """
        space = self.space
        q = self.q
        self.lform = LinearForm(space)
        method = self.assemble_method
        self.SSI = ScalarSourceIntegrator(q = q, method=method)
        self.lform.add_integrator(self.SSI)

    @variantmethod('forward')
    def update_coef(self,t1 = 0.0,uh0 = None,phi0 = None, mesh_vector = None):
        """
        Update the coefficients for the bilinear and linear forms.
        
        Parameters:
            t1 (float): The current time in the simulation.
            uh0 (Function, optional): Previous solution for the vector field variable.
            phi0 (Function, optional): Previous condition for the phase field variable.
            mesh_vector (Function, optional): Mesh vector for the Previous time.
        """
        pde = self.pde
        space = self.space
        eta = pde.eta
        gamma = pde.gamma
        dt = self.dt
        theta = self.theta # The Lagrange multiplier (default is 0).

        if uh0 is None:
            uh0 = self.uspace.interpolate(lambda p: pde.velocity_field(p, t=t1))
        if phi0 is None:
            phi0 = self.phi0
        if mesh_vector is None:
            mesh_vector = self.mspace.function()
        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
        else:
            phi_force = space.function()
        
        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            f_force = space.interpolate(phi_force)
            fphi = (phi_val**3 - phi_val) / (eta**2)
 
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * f_force(bcs, index)

            mesh_result = mesh_vector(bcs, index) - uh0(bcs, index)
            result += dt*bm.einsum('cqi, cqi->cq', mesh_result, phi0.grad_value(bcs, index))
            result += dt*gamma*theta
            return result
        self.SSI.source = source

        self.logger.info(f"Coefficients updated at time {t1} with theta = {theta} .")

    @update_coef.register('backward')
    def update_coef(self, t1 = 0.0, uh0 = None, phi0 = None, mesh_vector = None):
        """
        Update the coefficients for the bilinear and linear forms.
        
        Parameters:
            t1 (float): The current time in the simulation.
            uh0 (Function, optional): Previous solution for the vector field variable.
            phi0 (Function, optional): Previous condition for the phase field variable.
            mesh_vector (Function, optional): Mesh vector for the Previous time.
        """
        pde = self.pde
        space = self.space
        eta = pde.eta
        gamma = pde.gamma
        bcs = self.bcs
        dt = self.dt
        theta = self.theta # The Lagrange multiplier (default is 0).

        if uh0 is None:
            uh0 = self.uspace.interpolate(lambda p: pde.velocity_field(p, t=t1))
        if phi0 is None:
            phi0 = self.phi0
        if mesh_vector is None:
            mesh_vector = self.mspace.function()
        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
        else:
            phi_force = space.function()

        self.SCI.coef = dt*(uh0(bcs))
        
        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            f_force = space.interpolate(phi_force)
            fphi = (phi_val**3 - phi_val) / (eta**2)

            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * f_force(bcs, index)
            return result
        self.SSI.source = source

        self.logger.info(f"Coefficients updated at time {t1} with theta = {theta} .")
    
    @variantmethod('explicit')
    def lagrange_multi_solver(self,t1 = 0.0, phi0 = None, uh0 = None, mesh_vector = None):
        """
        Explicitly solve the Lagrange multiplier for the Allen-Cahn equation.
        """
        pde = self.pde
        space = self.space
        area = pde.area
        gamma = pde.gamma
        eta = pde.eta
        alpha = 1/self.dt + gamma/eta**2
        NN = self.mesh.number_of_nodes()
        GD = self.mesh.GD
        if uh0 is None:
            uh0 = self.uspace.interpolate(lambda p: pde.velocity_field(p, t=t1))
        if phi0 is None:
            phi0 = self.phi0
        if mesh_vector is None:
            mesh_vector = self.mspace.function()
        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
        else:
            phi_force = space.function()

        @barycentric
        def G(bcs):
            phi_val = phi0(bcs)
            f_force = space.interpolate(phi_force)
            fphi = (phi_val**3 - phi_val) / (eta**2)
            grad_phi0 = phi0.grad_value(bcs)
            cell_grad = bm.mean(grad_phi0, axis=1)
            vertex_grad = bm.zeros((NN, GD))
            vertex_count = bm.zeros(NN)
            cell2dof = space.cell_to_dof()
            vertex_grad = bm.index_add(vertex_grad, cell2dof, cell_grad[:, None])
            vertex_count = bm.index_add(vertex_count, cell2dof, 1)
            vertex_grad /= vertex_count[:, None] 

            grad_basis = space.grad_basis(bcs)
            laplace_phi0 = bm.einsum('cqid, cid->cq', grad_basis, vertex_grad[cell2dof])
            result = gamma * fphi
            result += f_force(bcs)
            result += gamma* laplace_phi0

            mesh_result = mesh_vector(bcs) - uh0(bcs)
            result += bm.einsum('cqi, cqi->cq', mesh_result, grad_phi0)
            return result
        
        node0 = self.mspace.function(self.node0.T.flatten())
        bcs,ws = self.bcs, self.ws
        cm = self.mesh.entity_measure('cell')
        J = bm.abs(node0.grad_value(bcs))
        value = alpha*(bm.linalg.det(J)-1)*phi0(bcs) - G(bcs)
        self.theta = 1/(gamma*area)*bm.einsum('cq ,q ,c ->',value , ws,cm)

    @variantmethod("direct")
    def solve(self):
        """
        Solve the weak formulation of the Allen-Cahn equation.
        
        This method assembles the system matrix and right-hand side vector, applies boundary conditions,
        and solves the linear system using a direct solver.
        """  
        A = self.bform.assembly()
        b = self.lform.assembly()

        phi_new = spsolve(A, b,solver= 'scipy')
        return phi_new
    
    @solve.register("cg")
    def solve(self, atol: float = 1e-14,rtol:float = 1e-14, maxit: int = 1000):
        """
        Solve the weak formulation of the Allen-Cahn equation using a conjugate gradient method.
        
        Parameters:
            tol (float): The tolerance for convergence.
            maxiter (int): The maximum number of iterations.
        
        Returns:
            phi_new (TensorLike): The updated phase field variable.
        """
        bform = self.bform
        lform = self.lform
        
        A = bform.assembly()
        b = lform.assembly()

        phi_new,info = cg(A, b,maxit=maxit,atol=atol, rtol=rtol, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return phi_new
    
    @variantmethod('fixed_mesh')
    def time_step(self):
        """
        Perform a time step for the Allen-Cahn phase field model.
        
        This method updates the phase field variable using the Lagrange multiplier method,
        solves the linear system with the mesh fixed.
        """
        self.lagrange_multi_solver(t1=self.t, phi0=self.phi0)
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.velocity_field(p, t=self.t))
        print(bm.max(self.uh0))
        self.update_coef(self.t,phi0=self.phi0, uh0=self.uh0)
        phi_new = self.solve()
        
        self.phi[:] = phi_new
        self.phi0[:] = self.phi[:]

    @time_step.register('moving_mesh')
    def time_step(self):
        """
        Perform a time step for the Allen-Cahn phase field model with a moving mesh.
        
        This method updates the phase field variable using the Lagrange multiplier method,
        solves the linear system, and moves the mesh based on the updated phase field variable.
        """
        mm = self.mm
        mspace = self.mspace
        mm.run()
        if self.t-self.dt == 0.0:
            self.node0 = self.mesh.node.copy()
            self.phi0[:] = self.space.interpolate(lambda p:self.pde.init_solution(p, t=self.t-self.dt))

        mesh_vector = mspace.function(((self.mesh.node - self.node0)/self.dt).T.flatten())
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.velocity_field(p, t=self.dt))
        self.lagrange_multi_solver(t1=self.t, phi0=self.phi0, uh0=self.uh0,mesh_vector=mesh_vector)
        self.update_coef(t1=self.t, uh0=self.uh0, phi0=self.phi0)
        phi_new = self.solve()
        self.phi[:] = phi_new
        self.phi0[:] = self.phi[:]
        self.node0 = self.mesh.node.copy()

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
        #f[tag0] = 2/eta**2  * (phi[tag0] - 1)
        #f[tag1] = (phi[tag1]**3 - phi[tag1]) / (eta**2)
        #f[tag2] = 2/eta**2 * (phi[tag2] + 1)
        f[:] = (phi[:]**3 - phi[:]) / (eta**2)
        return f
    
    def run(self):
        """
        Run the time-stepping loop for the Allen-Cahn phase field model.
        
        This method performs the time-stepping loop, updating the phase field variable
        and moving the mesh if necessary.
        """
        import matplotlib.pyplot as plt
        from ..mmesh.tool import linear_surfploter
        from ..functionspace import  TensorFunctionSpace
        self.logger.info("Starting time-stepping loop.")
        smspace = LagrangeFESpace(self.mesh, p=self.p)
        self.mspace = TensorFunctionSpace(smspace, (self.mesh.GD,-1))
        self.node0 = self.mesh.node.copy()
        i = 0
        while self.t < self.pde.duration()[1]:
            self.t += self.dt
            i += 1
            self.time_step()
            # phi_exact = self.space.interpolate(lambda p: self.pde.solution(p, self.t))
            # phi_error = self.mesh.error(self.phi, phi_exact, power=2)
            self.logger.info(f"Time: {self.t:.4f}")
            self.mesh.nodedata['interface'] = self.phi
            fname = './' + 'test3_ac'+ str(i).zfill(10) + '.vtu'
            self.mesh.to_vtk(fname=fname)
            if i > 10000:
                self.logger.info(f"Stopping after {i} iterations.")
                break
            # if self.t > 100:
            #     fig = plt.figure(figsize=(8, 8))
            #     ax = fig.add_subplot(111, projection='3d')
            #     linear_surfploter(ax, self.mesh, self.phi, scat_node=False)
            #     plt.show()

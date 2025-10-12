from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEModelManager, ComputationalModel
from ..model.allen_cahn import AllenCahnPDEDataProtocol
from ..decorator import variantmethod,barycentric
from ..mesh import Mesh
# FEM imports
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm,BlockForm
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
    def __init__(self,options: Optional[dict] = None):
        super().__init__(pbar_log=options['pbar_log'], 
                         log_level=options['log_level'])
        self.options = options
        self.assemble_method = None
        self.pdm = PDEModelManager("allen_cahn")
        
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'])
        self.set_space_degree(options['space_degree'])
        self.set_quadrature(options['quadrature'])
        self.set_assemble_method(options['assemble_method'])
        self.set_time_step(options['time_step'])
        self.set_function_space()
        self.set_velocity_function_space(up=options['up'])
        self.set_initial_condition(t=0.0)
        self.set_linear_system[options['time_strategy']]()
        self.solve.set(options['solve'])
        
        self.set_lagrange_multiplier.set(options['lagrange_multiplier'])
        self.lagrange_multiplier.set(options['lagrange_multiplier'])
        self.set_lagrange_multiplier()

    def set_pde(self,pde: Union[AllenCahnPDEDataProtocol,str] = 'circle_interface'):
        """
        Set the PDE data for the model.
        
        Parameters:
            pde (str or AcCircleData2D): The name of the PDE or an instance of AcCircleData2D.
        """
        if isinstance(pde, str):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde
        
    def set_init_mesh(self,mesh: Union[Mesh,str] = 'tri', **kwargs):
        """
        Set the initial mesh for the model.
        Parameters:
            mesh (Mesh or str): The mesh object or a string identifier for the mesh type.
            **kwargs: Additional keyword arguments for mesh creation.
        """
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
        
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
        Set the index of quadrature formula for numerical integration.
        
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

    def set_velocity_function_space(self,up: int = 2):
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
        self.phi0[:] = space.interpolate(lambda p:pde.init_condition(p, t=t))
    
    @variantmethod('implicit')
    def set_lagrange_multiplier(self):
        """
        Set the Lagrange multiplier for the weak formulation of the Allen-Cahn equation.
        This method directly introduces the Lagrange multiplier into the left-hand matrix for implicit solving.
        """
        from ..sparse import COOTensor
        LagLinearForm = LinearForm(self.space)
        Lag_SSI = ScalarSourceIntegrator(source=1, q=self.q)
        LagLinearForm.add_integrator(Lag_SSI)
        LagA = LagLinearForm.assembly()
        A0 = -self.dt * self.pde.gamma() * bm.ones(self.space.number_of_global_dofs())
        
        self.A0 = COOTensor(bm.array([bm.arange(len(A0), dtype=bm.int32), 
                                 bm.zeros(len(A0), dtype=bm.int32)]), A0, spshape=(len(A0), 1))
        self.A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA,
                                spshape=(1, len(LagA)))
        self.b0 = bm.array([self.mesh.integral(self.pde.init_condition)])
    
    @set_lagrange_multiplier.register('explicit')
    def set_lagrange_multiplier(self):
        """
        Set the Lagrange multiplier for the weak formulation of the Allen-Cahn equation.
        This method differs from the implicit scheme as it requires explicitly 
        solving the Lagrange multiplier before assembling the right-hand side
        """
        self.LagLinearForm = LinearForm(self.space)
        self.Lag_SSI = ScalarSourceIntegrator(q=self.q)
        self.LagLinearForm.add_integrator(self.Lag_SSI)
          
    @variantmethod('implicit')
    def lagrange_multiplier(self,A,b, uh0=None, phi_force=None):
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
        b  = bm.concatenate([b, b0], axis=0)
        return A, b
    
    @lagrange_multiplier.register('explicit')
    def lagrange_multiplier(self,A,b, uh0=None, phi_force=None):
        """
        Explicitly add the Lagrange multiplier terms to right-hand side vector.
        This method computes the Lagrange multiplier terms based on the current phase field variable and mesh movement.
        
        Parameters:
            A (CSRTensor): The original matrix from the bilinear form.
            b (TensorLike): The original right-hand side vector from the linear form.
            uh0 (Function): The previous velocity field variable.
            phi_force (Function): The force term for the phase field variable.
        Returns:
            A (CSRTensor): Without changing the matrix, only the right-hand side vector is extended.
            b (TensorLike): The extended right-hand side vector with Lagrange multiplier terms.
        """
        space = self.space
        bcs = self.bcs
        pde = self.pde
        gamma = pde.gamma()
        dt = self.dt
        eta = pde.eta
        phi0 = self.phi0
        rm = space.mesh.reference_cell_measure()
        @barycentric
        def G(bcs):
            phi_val = phi0(bcs)
            fphi = self.pde.nonlinear_source(phi_val)
    
            result = gamma * fphi
            result += phi_force(bcs)
            result += gamma* self.laplace_phi(bcs)
            
            mesh_result = self.move_vector(bcs) - uh0(bcs)
            result += bm.einsum('cqi, cqi->cq', mesh_result, phi0.grad_value(bcs))
            return result
        node0 = self.mspace.function(self.node0.T.flatten())
        J = node0.grad_value(bcs)
        value = (1/dt + gamma/eta**2)*(bm.abs(bm.linalg.det(J))-1)*phi0(bcs) - G(bcs)
        theta = 1/(gamma*pde.area)*bm.einsum('cq ,q ,cq ->',value,self.ws*rm,self.d)
        self.Lag_SSI.source = dt*gamma*theta
        b0 = self.LagLinearForm.assembly()
        b += b0
        return A,b
        
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
        self.mm.config.pde = self.pde
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
        self.move_vector = self.mspace.function()
        self.time_step.set('moving_mesh')
        if hasattr(self.space.mesh, 'jacobi_matrix'):
            self._Jacobi = self.space.mesh.jacobi_matrix
        else:
            self._Jacobi = self.space.mesh.jacobian_matrix
    
    @variantmethod('forward')
    def set_linear_system(self):
        """
        Set the linear system for the weak formulation of the Allen-Cahn equation.
        
        This method initializes the bilinear and linear forms for the forward time-stepping scheme.
        """
        space = self.space
        gamma = self.pde.gamma()
        eta = self.pde.eta
        dt = self.dt
        q = self.q
        method = self.assemble_method
        self.bform = BilinearForm(space)
        self.SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q, method=method)
        self.SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        self.bform.add_integrator(self.SMI,self.SDI)
        self.set_linear_form()
        
        self.logger.info("Linear system set with bilinear and linear forms.")
        self.logger.info("Bilinear form forward set with diffusion and mass terms.")
        
        self.assembly.set('forward')
        self.logger.info("Assembly method set to forward.")
        
    @set_linear_system.register('backward')
    def set_linear_system(self):
        """
        Set the bilinear form for the weak formulation of the Allen-Cahn equation.
        
        This method defines the bilinear form using diffusion, convection, and mass integrators.
        """
        space = self.space
        gamma = self.pde.gamma()
        eta = self.pde.eta
        dt = self.dt
        q = self.q
        method = self.assemble_method
        self.bform = BilinearForm(space)
        self.SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q,method=method)
        self.SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        self.SCI = ScalarConvectionIntegrator(q=q, method=method)
        self.bform.add_integrator(self.SMI,self.SDI, self.SCI)
        self.set_linear_form()
        
        self.logger.info("Linear system set with bilinear and linear forms.")
        self.logger.info("Bilinear form backward set with diffusion,convection and mass terms.")
        
        self.assembly.set('backward')
        self.logger.info("Assembly method set to backward.")

    @set_linear_system.register('moving_mesh')
    def set_linear_system(self):
        """
        Set the linear system for the weak formulation of the Allen-Cahn equation with moving mesh.
        """
        space = self.space
        gamma = self.pde.gamma()
        eta = self.pde.eta
        dt = self.dt
        q = self.q
        method = self.assemble_method
        self.bform = BilinearForm(space)
        self.bform_exp = BilinearForm(space)
        self.SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q, method=method)
        self.SMI_exp = ScalarMassIntegrator(coef=1.0, q=q, method=method)
        self.SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        self.bform.add_integrator(self.SMI,self.SDI)
        self.bform_exp.add_integrator(self.SMI_exp)
        self.set_linear_form()
        
        self.logger.info("Linear system set with bilinear and linear forms.")
        self.logger.info("Bilinear form moving_mesh set with diffusion and mass terms.")
        
        self.assembly.set('moving_mesh')
        if self.options['mm_param'] is None: 
            self.set_move_mesher()
        else:
            valid_keys = {'mmesher', 'beta', 'tau', 'tmax', 'alpha', 'moltimes', 'monitor', 'mol_meth'}
            filtered_params = {k: v for k, v in self.options['mm_param'].items() if k in valid_keys}
            self.set_move_mesher(**filtered_params)
        self.logger.info("Assembly method set to moving_mesh.")
        
    @set_linear_system.register('moving_mesh_SDIRK')
    def set_linear_system(self):
        """
        Set the linear system for the weak formulation of the Allen-Cahn equation with moving mesh.
        """
        space = self.space
        q = self.q
        method = self.assemble_method
        self.bform0 = BilinearForm(space)
        self.bform1 = BilinearForm(space)
        self.SMI = ScalarMassIntegrator(q=q, method=method)
        self.SDI = ScalarDiffusionIntegrator(q=q, method=method)
        self.SCI = ScalarConvectionIntegrator(q=q, method=method)
        self.bform0.add_integrator(self.SDI, self.SCI)
        self.bform1.add_integrator(self.SMI)
        self.set_linear_form()
        
        self.logger.info("Linear system set with bilinear and linear forms.")
        self.logger.info("Bilinear form moving_mesh set with diffusion and mass terms.")
        
        if self.options['mm_param'] is None: 
            self.set_move_mesher()
        else:
            valid_keys = {'mmesher', 'beta', 'tau', 'tmax', 'alpha', 'moltimes', 'monitor', 'mol_meth'}
            filtered_params = {k: v for k, v in self.options['mm_param'].items() if k in valid_keys}
            self.set_move_mesher(**filtered_params)
        self.logger.info("Assembly method set to moving_mesh.")
    
    def laplace_recovery(self,phi0):
        """
        Compute the Laplacian recovery for the phase field variable.
        
        This method uses the L2 projector compute the Laplacian.
        
        Parameters:
            phi0 (Function): The phase field variable function.
        
        Returns:
            Function: The recovered Laplacian of the phase field variable.
        """
        space = self.space
        bcs = self.bcs
        grad_phi0 = phi0.grad_value(bcs)
        grad_basis = space.grad_basis(bcs)
        rm = space.mesh.reference_cell_measure()
        J = self._Jacobi(bcs)
        G = bm.permute_dims(J, (0, 1, 3,2))@ J
        self.d = bm.sqrt(bm.linalg.det(G))
        b_local = bm.einsum('q,cq,cqid, cqd->ci',self.ws*rm,self.d ,grad_basis,-grad_phi0)
        F = bm.zeros(space.number_of_global_dofs(), dtype=bm.float64)
        F = bm.index_add(F, space.cell_to_dof(), b_local)
        M = self.bform_exp.assembly()
        laplace_phi = space.function(self.solve(M, F))
        return laplace_phi
       
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
    def assembly(self,t1 = 0.0,uh0 = None,phi0 = None):
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
        gamma = pde.gamma()
        dt = self.dt

        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
            phi_force = space.interpolate(phi_force)
        else:
            phi_force = space.function()
        
        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            fphi = pde.nonlinear_source(phi_val)
 
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * phi_force(bcs, index)

            mesh_result = -uh0(bcs, index)
            result += dt*bm.einsum('cqi, cqi->cq', mesh_result, phi0.grad_value(bcs, index))
            return result
        self.SSI.source = source

        self.logger.info(f"Coefficients updated at time {t1}.")
        A = self.bform.assembly()
        b = self.lform.assembly()
        return A, b

    @assembly.register('backward')
    def assembly(self, t1 = 0.0, uh0 = None, phi0 = None):
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
        gamma = pde.gamma()
        bcs = self.bcs
        dt = self.dt

        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
            phi_force = space.interpolate(phi_force)
        else:
            phi_force = space.function()

        self.SCI.coef = dt*(uh0(bcs))
        
        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            fphi = pde.nonlinear_source(phi_val)
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * phi_force(bcs, index)
            return result
        self.SSI.source = source

        self.logger.info(f"Coefficients updated at time {t1} .")
        A = self.bform.assembly()
        b = self.lform.assembly()
        return A, b
    
    @assembly.register('moving_mesh')
    def assembly(self, t1 = 0.0, uh0 = None, phi0 = None, move_vector = None):
        """
        Assemble the system matrix and right-hand side vector for the Allen-Cahn equation.
        
        Parameters:
            t1 (float): The current time in the simulation.
            uh0 (Function, optional): Previous solution for the vector field variable.
            phi0 (Function, optional): Previous condition for the phase field variable.
            mesh_vector (Function, optional): Mesh vector for the Previous time.
        
        Returns:
            A (SparseMatrix): The system matrix.
            b (TensorLike): The right-hand side vector.
        """
        pde = self.pde
        space = self.space
        eta = pde.eta
        gamma = pde.gamma()
        dt = self.dt

        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
            self.phi_force = space.interpolate(phi_force)
        else:
            self.phi_force = space.function()

        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            fphi = pde.nonlinear_source(phi_val)
            result = -gamma * dt * fphi
            result += gamma * dt * self.laplace_phi(bcs, index)
            result += dt * self.phi_force(bcs, index)
            
            mesh_result = move_vector(bcs,index) - uh0(bcs, index)
            result += dt*bm.einsum('cqi, cqi->cq', mesh_result, phi0.grad_value(bcs, index))
            return result
        self.SSI.source = source

        self.logger.info(f"Coefficients updated at time {t1}.")
        A = self.bform.assembly()
        b = self.lform.assembly()
        return A, b

    @variantmethod("direct")
    def solve(self,A,b):
        """
        Solve the weak formulation of the Allen-Cahn equation.
        
        Parameters:
            A (SparseMatrix): The system matrix.
            b (TensorLike): The right-hand side vector.
            
        This method assembles the system matrix and right-hand side vector, applies boundary conditions,
        and solves the linear system using a direct solver.
        """  
        phi_new = spsolve(A, b,solver= 'scipy')
        return phi_new
    
    @solve.register("cg")
    def solve(self,A,b, atol: float = 1e-14,rtol:float = 1e-14, maxit: int = 1000):
        """
        Solve the weak formulation of the Allen-Cahn equation using a conjugate gradient method.
        
        Parameters:
            A (SparseMatrix): The system matrix.
            b (TensorLike): The right-hand side vector.
            tol (float): The tolerance for convergence.
            maxit (int): The maximum number of iterations.
        
        Returns:
            phi_new (TensorLike): The updated phase field variable.
        """
        phi_new,info = cg(A, b,maxit=maxit,atol=atol, rtol=rtol, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return phi_new
    
    @variantmethod
    def time_step(self):
        """
        Perform a time step for the Allen-Cahn phase field model.
        
        This method updates the phase field variable using the Lagrange multiplier method,
        solves the linear system with the mesh fixed.
        """
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.velocity_field(p, t=self.t))
        A,b = self.assembly(t1 = self.t,uh0=self.uh0,phi0=self.phi0)

        A,b = self.lagrange_multiplier(A, b)
        phi_new = self.solve(A,b)[:-1]
        
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
        mm.run()
        if self.t-self.dt == 0.0:
            mm.set_interpolation_method('solution')
            self.node0 = self.mesh.node.copy()
            self.phi0[:] = self.space.interpolate(lambda p:self.pde.init_condition(p, t=self.t-self.dt))
            mm.set_interpolation_method('linear')
        self.move_vector[:] = ((self.mesh.node - self.node0)/self.dt).T.flatten()
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.velocity_field(p, t=self.dt))
        
        self.laplace_phi = self.laplace_recovery(self.phi0)
        A,b = self.assembly(t1 = self.t,uh0 = self.uh0,phi0=self.phi0,move_vector = self.move_vector)
        A,b = self.lagrange_multiplier(A, b, uh0=self.uh0, phi_force=self.phi_force)

        phi_e = self.solve(A,b)
        phi_new = self.phi0 + phi_e

        self.phi[:] = phi_new
        self.phi0[:] = self.phi[:]
        self.node0 = self.mesh.node.copy()
        self.mm.instance.uh = phi_new.copy()
    
    @time_step.register('moving_mesh_SDIRK')
    def time_step(self):
        """
        Perform a time step for the Allen-Cahn phase field model using the SDIRK method.
        
        This method updates the phase field variable using the SDIRK method, which is suitable for stiff problems.
        """
        sub_steps = 30
        r = 1- bm.sqrt(2)/2
        tau1 = r
        tau2 = 1
        a11 = r
        a21 = 1-r
        a22 = r
        b1 = 1-r
        b2 = r
        mm = self.mm
        mm.run()
        dt = self.dt
        if self.t-dt == 0.0:
            sub_steps = 30
            self.node0 = self.mesh.node.copy()
            self.phi0[:] = self.space.interpolate(lambda p:self.pde.init_condition(p, t=self.t-self.dt))
        else:
            sub_steps = 3
        
        delta = dt/(sub_steps)    
        v = (self.mesh.node - self.node0)/self.dt
        self.move_vector[:] = (v).T.flatten()
        for i in range(sub_steps):
            t_hat = self.t + (i)*delta
            self.mesh.node = self.node0 + tau1*delta * v
            M = self.bform1.assembly()
            self.SDI.coef = delta * self.pde.gamma() *a11
            self.SCI.coef = -delta * self.move_vector *a11
            def source(bcs,index):
                phi_val = self.phi0(bcs, index)
                f_val = self.pde.nonlinear_source(phi_val)
                result =  phi_val
                result -= a11* delta * self.pde.gamma() * f_val
                return result
            
            self.SSI.source = source
            A = self.bform0.assembly() 
            A += M
            b = self.lform.assembly()
            
            self.set_lagrange_multiplier()
            A,b = self.lagrange_multiplier(A, b)
            phi1 = self.solve(A, b)[:-1]
            
            self.mesh.node = self.node0 + delta * v
            phi1 = self.space.function(phi1)
            k1 = (phi1 - self.phi0)/(tau1*delta)
            self.SDI.coef = self.pde.gamma()*delta*a22
            self.SCI.coef = -delta * self.move_vector*a22
            def source(bcs,index):
                result = delta*a21*k1.value(bcs, index)
                result += self.phi0(bcs, index)
                
                phi_val = self.phi0(bcs, index)
                f_val = self.pde.nonlinear_source(phi_val)
                result -= a22* delta * self.pde.gamma() * f_val
                return result
            self.SSI.source = source
            A = self.bform0.assembly()
            M = self.bform1.assembly()
            A += M
            b = self.lform.assembly()
            self.set_lagrange_multiplier()
            A,b = self.lagrange_multiplier(A, b)
            phi2 = self.solve(A, b)[:-1]
            
            k2 = (phi2 - self.phi0 - delta*a21*k1)/(a22*delta)
            self.phi[:] = self.phi0 + delta*b1*k1 + delta*b2*k2
            self.phi0[:] = self.phi[:]
            self.node0 = self.mesh.node.copy()
        self.mm.instance.uh = self.phi.copy()
                   
    def run(self,save_vtu_enabled: bool = False,error_estimate_enabled: bool = False):
        """
        Run the time-stepping loop for the Allen-Cahn phase field model.
        
        This method performs the time-stepping loop, updating the phase field variable
        and moving the mesh if necessary.
        """
        self.logger.info("Starting time-stepping loop.")
        self.error_estimate_enabled = error_estimate_enabled
        step = 0
        while self.t < self.pde.duration()[1]:
            self.t += self.dt
            self.time_step()
            self.logger.info(f"Time: {self.t:.5f}")
            
            if save_vtu_enabled:
                self.save_vtu(step)

            if self.error_estimate_enabled:
                error = self.error_estimate()
                
            step += 1

    def save_vtu(self, step: int):
        """
        Save the current state of the simulation to a .vtu file.

        Parameters:
            step (int): The current time step index.
        """
        self.mesh.nodedata['interface'] = self.phi
        self.mesh.nodedata['velocity'] = self.uh0.reshape(self.mesh.GD,-1).T
        fname = './' + 'test_ac' + str(step).zfill(10) + '.vtu'
        self.mesh.to_vtk(fname=fname)
        self.logger.info(f"VTU file saved: {fname}")
        
    def error_estimate(self):
        """
        Estimate the error between the computed phase field variable and the exact solution.
        
        Parameters:
            exact_solution (TensorLike): The exact solution for comparison.
        
        Returns:
            error (float): The estimated error.
        """
        if not hasattr(self.pde, 'solution'):
            self.logger.warning("pde.solution is not defined. Disabling error estimation.")
            self.error_estimate_enabled = False
            return 0.0
        
        pde = self.pde
        error = self.mesh.error(self.phi,
                                lambda p: pde.solution(p, t=self.t),)
        
        self.logger.info(f"Error estimate: {error:.4e}")
        return error
        
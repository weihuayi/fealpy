from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEDataManager, ComputationalModel
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
        self.is_Lag_muliplier = False
        self.pdm = PDEDataManager("allen_cahn")
        
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
        
        if options['is_volume_conservation']:
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
        self.phi0[:] = space.interpolate(lambda p:pde.init_solution(p, t=t))
    
    def set_lagrange_multiplier(self):
        """
        Set the Lagrange multiplier for the weak formulation of the Allen-Cahn equation.
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
        self.b0 = bm.array([self.mesh.integral(self.pde.init_solution)])
        self.is_Lag_muliplier = True
        
    def lagrange_multiplier(self,A,b):
        A0 = self.A0
        A1 = self.A1
        b0 = self.b0
        A = BlockForm([[A, A0], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b  = bm.concatenate([b, b0], axis=0)
        return A, b
        
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
        self.mm.config.is_pre = True
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
        self.bform.add_integrator(self.SMI)
        self.bform.add_integrator(self.SDI)
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
        self.bform.add_integrator(self.SMI)
        self.bform.add_integrator(self.SDI)
        self.bform.add_integrator(self.SCI)
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
        self.SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=q, method=method)
        self.SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q, method=method)
        # self.SCI = ScalarConvectionIntegrator(q=q, method=method)
        self.bform.add_integrator(self.SMI)
        self.bform.add_integrator(self.SDI)
        # self.bform.add_integrator(self.SCI)
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
        theta = self.theta # The Lagrange multiplier (default is 0).

        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
        else:
            phi_force = space.function()
        
        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            f_force = space.interpolate(phi_force)
            fphi = pde.nonlinear_source(phi_val)
 
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * f_force(bcs, index)

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
        theta = self.theta # The Lagrange multiplier (default is 0).

        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
        else:
            phi_force = space.function()

        self.SCI.coef = dt*(uh0(bcs))
        
        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            f_force = space.interpolate(phi_force)
            fphi = pde.nonlinear_source(phi_val)
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * f_force(bcs, index)
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
        bcs = self.bcs
        dt = self.dt
        theta = self.theta # The Lagrange multiplier (default is 0).

        if hasattr(pde , 'phi_force'):
            phi_force = lambda p: pde.phi_force(p, t1)
        else:
            phi_force = space.function()

        @barycentric
        def source(bcs, index):
            phi_val = phi0(bcs, index)
            # f_force = space.interpolate(phi_force)
            fphi = pde.nonlinear_source(phi_val)
 
            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            # result += dt * f_force(bcs, index)

            mesh_result = move_vector(bcs,index) - uh0(bcs, index)
            result += dt*bm.einsum('cqi, cqi->cq', mesh_result, phi0.grad_value(bcs, index))
            return result
        self.SSI.source = source

        self.logger.info(f"Coefficients updated at time {t1}.")
        A = self.bform.assembly()
        b = self.lform.assembly()
        return A, b
    
    @variantmethod('explicit')
    def lagrange_multi_solver(self,t1 = 0.0, phi0 = None, uh0 = None, mesh_vector = None):
        """
        Explicitly solve the Lagrange multiplier for the Allen-Cahn equation.
        """
        pde = self.pde
        space = self.space
        area = pde.area
        gamma = pde.gamma()
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
            fphi = pde.nonlinear_source(phi_val)
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
        if self.is_Lag_muliplier:
            A,b = self.lagrange_multiplier(A, b)
            phi_new = self.solve(A,b)[:-1]
        else:
            phi_new = self.solve(A,b)
        
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
            self.node0 = self.mesh.node.copy()
            self.phi0[:] = self.space.interpolate(lambda p:self.pde.init_solution(p, t=self.t-self.dt))

        self.move_vector[:] = ((self.mesh.node - self.node0)/self.dt).T.flatten()
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.velocity_field(p, t=self.dt))

        A,b = self.assembly(t1 = self.t,uh0=self.uh0,phi0=self.phi0,move_vector = self.move_vector)
        if self.is_Lag_muliplier:
            self.set_lagrange_multiplier()
            A,b = self.lagrange_multiplier(A, b)
            phi_new = self.solve(A,b)[:-1]
        else:
            phi_new = self.solve(A,b)

        self.phi[:] = phi_new
        self.phi0[:] = self.phi[:]
        self.node0 = self.mesh.node.copy()
        self.mm.instance.uh = phi_new
    
    def run(self,save_vtu_enabled: bool = False,error_estimate_enabled: bool = False):
        """
        Run the time-stepping loop for the Allen-Cahn phase field model.
        
        This method performs the time-stepping loop, updating the phase field variable
        and moving the mesh if necessary.
        """
        self.logger.info("Starting time-stepping loop.")
        
        step = 0
        while self.t < self.pde.duration()[1]:
            self.t += self.dt
            self.time_step()
            self.logger.info(f"Time: {self.t:.5f}")
            
            if save_vtu_enabled:
                self.save_vtu(step)

            if error_estimate_enabled:
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
            error_estimate_enabled = False
            return 0.0
        
        pde = self.pde
        error = self.mesh.error(self.phi,
                                lambda p: pde.solution(p, t=self.t),)
        
        self.logger.info(f"Error estimate: {error:.4e}")
        return error
        
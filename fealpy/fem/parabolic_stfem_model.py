from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEModelManager, ComputationalModel
from ..model.parabolic import ParabolicPDEDataProtocol
from ..decorator import variantmethod,barycentric
from ..mesh import Mesh
# FEM imports
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import DirichletBC
from ..fem import SpaceTimeConvectionIntegrator
from ..fem import SpaceTimeSourceIntegrator
from ..fem import SpaceTimeDiffusionIntegrator
from ..fem import SpaceTimeMassIntegrator


class ParabolicSTFEMModel(ComputationalModel):
    """
    A class for solving parabolic PDEs using space-time finite element methods (STFEM).
    This class extends the PDEModelManager to handle parabolic problems with time-dependent solutions.
    The equation is typically of the form:
        u_t - div(K(x)∇u) + beta * ∇u + gamma*u = f
    where K is the diffusion coefficient, beta is the convection coefficient, 
    gamma is the reaction term, and f is the source term.
    
    We will transform this into a space-time problem:
        - div(D(x)∇_y(u)) + b * ∇_y(u) + gamma * u = f
    where y = (x, t) is the space-time variable 
    and D(x) is the diffusion tensor [[K(x),0],[0,0]]. b is the convection vector [beta, 1].
    """
    def __init__(self, options: Optional[dict] = None):
        super().__init__(pbar_log=options['pbar_log'], 
                         log_level=options['log_level'])
        self.options = options
        self.pdm = PDEModelManager("parabolic")
        
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], **options['mesh_size'])
        self.set_space_degree(options['space_degree'])
        self.set_quadrature(options['quadrature'])
        self.set_assemble_method(options['assemble_method'])
        self.solve.set(options['solver'])
        self.set_function_space()
        
    def set_pde(self,pde: Union[ParabolicPDEDataProtocol,int] = 1):
        """
        Set the PDE data for the model.
        
        Parameters:
            pde (str or ParabolicPDEDataProtocol): The name of the PDE or an instance of ParabolicPDEDataProtocol.
        """
        if isinstance(pde, int):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self,mesh: Union[Mesh,str] = 'uniform_tri', **kwargs):
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
        Set the function space for the finite element method.
        This method initializes the Lagrange finite element space based on the mesh and polynomial order.
        """
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.logger.info(f"Function space initialized with polynomial order {self.p}.")
        self.uh = self.space.function()
        
    def linear_system(self):
        """
        Set up the linear system for the parabolic PDE.
        This method initializes the bilinear and linear forms, and sets up the Dirichlet boundary conditions.
        """
        space = self.space
        pde = self.pde
        method = self.assemble_method
        q = self.q
        self.bform = BilinearForm(space)
        self.lform = LinearForm(space)
        self.logger.info("Linear system set with bilinear and linear forms.")
        STMI = SpaceTimeMassIntegrator(coef=pde.reaction_coef, q=q, method=method,conv_coef=pde.convection_coef)
        STDI = SpaceTimeDiffusionIntegrator(coef=pde.diffusion_coef, q=q, method=method, conv_coef=pde.convection_coef)
        STCI = SpaceTimeConvectionIntegrator(coef=pde.convection_coef, q=q, method=method)
        STSI = SpaceTimeSourceIntegrator(source=pde.source, q=q, method=method, conv_coef=pde.convection_coef)

        self.logger.info("Integrators for mass, diffusion, convection, and source terms created.")
        self.bform.add_integrator(STMI, STDI, STCI)
        self.logger.info("Bilinear form set with diffusion, convection and mass terms.")
        self.lform.add_integrator(STSI)
        self.logger.info("Linear form set with source term.")
        
        A = self.bform.assembly()
        self.logger.info(f"Linear system initialized with {A.shape} matrix.")
        b = self.lform.assembly()
        self.logger.info(f"Linear system initialized with {b.shape} right-hand side vector.")
        self.logger.info("Bilinear and linear forms assembled.")
        
        return A, b
    
    def apply_bc(self,A,b):
        """
        Apply Dirichlet boundary conditions to the linear system.
        This method sets up the Dirichlet boundary conditions based on the PDE data.
        """
        space = self.space
        pde = self.pde
        threshold_space = pde.is_dirichlet_boundary
        threshold_time = pde.is_init_boundary
        gd_space = pde.solution
        gd_time = pde.init_solution

        bc_space = DirichletBC(space=space, gd=gd_space, threshold=threshold_space)
        bc_time = DirichletBC(space=space, gd=gd_time, threshold=threshold_time)
        A, b = bc_space.apply(A, b)
        A, b = bc_time.apply(A, b)
        self.logger.info("Dirichlet boundary conditions applied with space and time.")
        return A, b
    
    @variantmethod("direct")
    def solve(self, A, b):
        """
        Solve the linear system Ax = b.
        
        Parameters:
            A (TensorLike): The coefficient matrix of the linear system.
            b (TensorLike): The right-hand side vector.
        
        Returns:
            TensorLike: The solution vector u.
        """
        from ..solver import spsolve
        uh = spsolve(A, b, solver='scipy')
        self.logger.info("Linear system solved.")
        return uh

    @solve.register("cg")
    def solve(self,A,b):
        """
        Solve the linear system using the conjugate gradient method.
        Parameters:
            A (TensorLike): The coefficient matrix of the linear system.
            b (TensorLike): The right-hand side vector.
        Returns:
            TensorLike: The solution vector u.
        """
        from ..solver import cg
        uh, info = cg(A, b, maxit=10000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return uh
    
    @variantmethod
    def run(self):
        A, b = self.linear_system()
        A, b = self.apply_bc(A, b)
        self.uh[:] = self.solve(A, b)
        self.logger.info("Run completed.")
        l2 , h1 = self.error()
        self.logger.info(f"L2 Error: {l2}, H1 Error: {h1}.")

    @run.register("uniform_refine")
    def run(self,maxit=4,plot_error=False):
        """
        Run the model with uniform mesh refinement.
        Parameters:
            maxit (int): The maximum number of refinement iterations.
        """
        error_matrix = bm.zeros((maxit+1,2), dtype=bm.float64)
        for i in range(maxit):
            A, b = self.linear_system()
            A, b = self.apply_bc(A, b)
            self.uh[:] = self.solve(A, b)
            l2,h1 = self.error()
            self.logger.info(f"{i}-th step with  L2 Error: {l2}, H1 Error: {h1}.")
            error_matrix[i,0] = l2
            error_matrix[i,1] = h1
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.mesh.uniform_refine()
                self.uh = self.space.function()
        if plot_error:
            self.show_order(error_matrix,maxit)

    def error(self):
        l2 = self.mesh.error(self.pde.solution, self.uh,q = self.q)
        h1 = self.mesh.error(self.pde.gradient, self.uh.grad_value,q = self.q)
        return l2, h1
    
    def show_solution(self):
        """
        Visualize the solution using the mesh's plotting capabilities.
        """
        import matplotlib.pyplot as plt
        from fealpy.mmesh.tool import linear_surfploter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        linear_surfploter(ax , self.mesh, self.uh)
        plt.show()
        self.logger.info("Solution visualized.")
        
    def show_order(self, error_matrix,maxit):
        """
        Show the convergence order of the solution.
        This method plots the convergence order based on the error matrix.
        """
        import matplotlib.pyplot as plt
        h0 = 0.1 * 1 / 2 ** (1 / 2)
        h = bm.array([h0 / (2 ** i) for i in range(maxit + 1)])
        log2h = bm.log2(h)

        l2_error = error_matrix[:, 0]
        h1_error = error_matrix[:, 1]
        print("L2_order:", (bm.log2(l2_error[1:] / l2_error[:-1])/ bm.log2(h[1:] / h[:-1]))[:-1])
        print("H1_order:", (bm.log2(h1_error[1:] / h1_error[:-1])/ bm.log2(h[1:] / h[:-1]))[:-1])
        plt.figure()
        # 横坐标用 -log2(h)，让曲线从左到右下降
        x = -log2h

        plt.plot(x, bm.log2(l2_error), 'o-', label='L2 error')
        plt.plot(x, bm.log2(h1_error), '^-', label='H1 error')

        # 理论参考线
        ref2 = (self.p+1) * (-x + x[0]) + 0.9*bm.log2(l2_error[0])
        plt.plot(x, ref2, 'k--', label=f'{self.p+1}nd order (ref)')

        ref1 = self.p * (-x + x[0]) + 0.9*bm.log2(h1_error[0])
        plt.plot(x, ref1, 'k-.', label=f'{self.p}nd order (ref)')

        plt.xlabel('|log2(h)|')
        plt.ylabel('log2(error)')
        plt.title('Convergence rate')
        plt.legend()
        plt.grid(True)
        # plt.gca().set_aspect('equal')
        plt.show()

    def slicing_error(self,t):
        node = self.mesh.node
        time_idx = bm.where(bm.abs(node[:, 1] - t) < 1e-8)[0]
        
        edge = self.mesh.edge
        time_edge = bm.all(bm.isin(edge, time_idx), axis=1)
        space_edge = bm.where(time_edge)[0]
            
        
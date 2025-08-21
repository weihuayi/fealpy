from typing import Optional, Union
from ..model import ComputationalModel
# from ..model.elliptic import EllipticPDEDataT

from ..functionspace import RaviartThomasFESpace2d
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarMassIntegrator, ScalarSourceIntegrator
from ..fem import DivIntegrator
from ..fem import BlockForm
from ..backend import backend_manager as bm
from ..fem import DirichletBC
from ..model import PDEModelManager
from ..decorator import variantmethod
from ..mesh import Mesh


class EllipticMixedFEMModel(ComputationalModel):
    """
    EllipticMixedFEMModel: Elliptic Mixed Finite Element Model

    This class implements a 2D elliptic PDE solver based on mixed finite element methods.
    It is suitable for elliptic problems requiring accurate flux approximation. The model supports custom PDE data,
    automatic mesh generation, linear system assembly, boundary condition handling, solution, and error analysis.

    Parameters:
        pde (EllipticPDEDataT, optional): The elliptic PDE data object, including coefficients, source terms, and boundary conditions.
            If None, a built-in example problem ('coscos') is used.
            
        mesh (Mesh, optional): The mesh object for the computational domain. If None, a uniform triangular mesh is created.

        p (int, optional): The polynomial degree of the finite element spaces. Default is
            0, which corresponds to piecewise constant elements for the primary variable and lowest order Raviart-Thomas elements for the flux.

        pbar_log (bool, optional): Whether to show a progress bar during the computation. Default is False.

        log_level (str, optional): The logging level for the model. Default is 'INFO'.

        apply_bc (str, optional): The type of boundary condition to apply. Default is 'dirichlet'.

        solve (str, optional): The type of solver to use for the linear system. Default is 'direct'.

        run (str, optional): The type of refinement strategy to use. Default is 'uniform'.

    Attributes:
        pde (EllipticPDEDataT): The elliptic PDE data object.

        mesh (Mesh): The mesh object for the computational domain.

        p (int): The polynomial degree of the finite element spaces.

        uspace (LagrangeFESpace): The Lagrange finite element space for the primary variable.

        pspace (RaviartThomasFESpace2d): The Raviart-Thomas finite element space for the flux.

        uh (ndarray): The numerical solution of the primary variable.

        ph (ndarray): The numerical solution of the flux.

        xh (ndarray): The concatenated solution vector containing both primary and flux variables.

        logger (Logger): The logger for the model, used for logging messages and progress.

    Methods:
        set_pde(pde: Optional[EllipticPDEDataT] = None) -> None
            Set the elliptic PDE data for the model. If None, a default example problem is used.

        set_init_mesh(mesh: Union[Mesh, str] = "uniform_tri", **kwargs) 
            Set the initial mesh for the model. If a string is provided, it is interpreted as a mesh type.

        set_space_degree(p: int)
            Set the polynomial degree of the finite element spaces.

        space(p: int = 0)
            Set the finite element spaces for the model, either Raviart-Thomas or Brezzi-Douglas-Marini.

        linear_system()
            Assemble the linear system for the elliptic mixed finite element model.

        apply_bc(A, b)
            Apply boundary conditions to the linear system.

        solve(A, b) 
            Solve the linear system using the specified solver type.
            
        run()
            Execute the simulation and return the results.
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'])
        self.set_space_degree(options['space_degree'])
        self.apply_bc.set(options['apply_bc'])
        self.solve.set(options['solve'])
        self.run.set(options['run'])
        

    def set_pde(self, pde=2)-> None:
        """
        Set the PDE data for the model.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('diffusion_reaction').get_example(pde)
        else:
            self.pde = pde

            
    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tri", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")
        
        
    def set_space_degree(self, p: int):
        """
        Set the polynomial degree of the finite element space.

        Args:
            p: The polynomial degree.
        """
        self.p = p
        
    @variantmethod("rt")
    def space(self, p: int = 0):
        """
        Set the finite element spaces for the model.
        
        Parameters
        p (int): The polynomial degree of the finite element spaces.
        """
        self.p = p
        self.pspace = RaviartThomasFESpace2d(self.mesh, p=p)
        self.uspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        
        return self.uspace, self.pspace
        
    @space.register("bdm")
    def space(self, p: int = 0):
        """
        Set the BDM finite element spaces for the model.
        
        Parameters:
        p (int): The polynomial degree of the BDM finite element spaces.
        """
        from ..functionspace import BrezziDouglasMariniFESpace
        self.p = p
        self.pspace = BrezziDouglasMariniFESpace(self.mesh, p=p)
        self.uspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        
        return self.uspace, self.pspace
        
    def linear_system(self):
        """
        Assemble the linear system for the elliptic mixed finite element model.
        """
        self.uspace, self.pspace = self.space(self.p) 

        uLDOF = self.uspace.number_of_local_dofs()
        uGDOF = self.uspace.number_of_global_dofs()
        pLDOF = self.pspace.number_of_local_dofs()
        pGDOF = self.pspace.number_of_global_dofs()
        self.logger.info(f"p_space: {self.pspace}, LDOF: {pLDOF}, GDOF: {pGDOF}")
        self.logger.info(f"u_space: {self.uspace}, LDOF: {uLDOF}, GDOF: {uGDOF}")
        self.uh = self.uspace.function()
        self.ph = self.pspace.function()
        self.xh = bm.zeros((uGDOF + pGDOF,), dtype=bm.float64)
        
        bform1 = BilinearForm(self.pspace)
        bform1.add_integrator(ScalarMassIntegrator(coef=self.pde.diffusion_coef_inv, q=3))


        bform2 = BilinearForm((self.uspace,self.pspace))
        bform2.add_integrator(DivIntegrator(coef=-1, q=3))

        bform3 = BilinearForm((self.uspace,self.pspace))
        bform3.add_integrator(DivIntegrator(coef=1, q=3))

        bform4 = BilinearForm(self.uspace)
        bform4.add_integrator(ScalarMassIntegrator(coef=2, q=3))

        M = BlockForm([[bform1,bform2],
                       [bform3.T,bform4]])
        A = M.assembly()
        
        lform = LinearForm(self.uspace)
        lform.add_integrator(ScalarSourceIntegrator(source=self.pde.source))
        F = lform.assembly()
        G = bm.zeros(pGDOF)
        b = bm.concatenate([G,F],axis=0)
        
        return A, b

    @variantmethod("neumann")
    def apply_bc(self, A, b):
        """
        Apply boundary conditions to the linear system.

        Parameters:
            A (ndarray): The system matrix.
            b (ndarray): The right-hand side vector.

        Returns:
            (A, b): The modified system matrix and right-hand side after applying boundary conditions.
        """

        self.uspace, self.pspace = self.space(self.p)
        uGDOF = self.uspace.number_of_global_dofs()
        ispBdof = self.pspace.is_boundary_dof()
        isyBdof = bm.zeros(uGDOF, dtype=bm.bool)
        isBdof = bm.concatenate([ispBdof,isyBdof],axis=0)
        fun = self.pspace.function()
        k,_ = self.pspace.set_dirichlet_bc(self.pde.grad_dirichlet,fun)
        k1 = bm.zeros(uGDOF, dtype=bm.float64)
        k = bm.concatenate([k,k1],axis=0)
        bc = DirichletBC(space=(self.pspace,self.uspace),gd=k, threshold=isBdof)
        A, b = bc.apply(A, b)
        
        return A, b
    
    @apply_bc.register("dirichlet")
    def apply_bc(self, A, b):
        """
        Apply Dirichlet boundary conditions to the linear system.

        This method modifies the system matrix and right-hand side vector to enforce Dirichlet boundary conditions
        for the mixed FEM system. It sets the corresponding rows and columns in the matrix and adjusts the right-hand side.

        Parameters:
            A (ndarray): The system matrix.
            b (ndarray): The right-hand side vector.

        Returns:
            (A, b): The modified system matrix and right-hand side after applying Dirichlet boundary conditions.
        """
        uspace, pspace = self.space(self.p)
        ugdof = uspace.number_of_global_dofs()
        G_apply = pspace.set_neumann_bc(self.pde.solution)
        F = bm.zeros(ugdof, dtype=bm.float64)
        b_apply = bm.concatenate([G_apply,F],axis=0)
        b = b - b_apply
        return A, b

    @variantmethod("direct")
    def solve(self, A, b):
        from fealpy.solver import spsolve
        return spsolve(A, b, solver='scipy')
    
    @solve.register('amg')
    def solve(self, A, F):
        pass

    @solve.register('cg')
    def solve(self, A, b):
        from fealpy.solver import cg 
        self.uh[:], info = cg(A, b, maxit=5000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
    
    @variantmethod('onestep')
    def run(self):
        """
        Execute the complete FEM solution process and return the numerical solutions of the primary and auxiliary variables.
        """
        A, b = self.linear_system()
        A, b = self.apply_bc(A, b)
        self.xh[:] = self.solve(A, b)
        self.ph[:] = self.xh[:self.pspace.number_of_global_dofs()]
        self.uh[:] = self.xh[self.pspace.number_of_global_dofs():]
        ul2, pl2 = self.postprocess()
        self.logger.info(f"u_L2 Error: {ul2},  p_L2 Error: {pl2}.")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        """
        Execute the complete FEM solution process with uniform mesh refinement.

        Parameters:
            maxit (int): The number of iterations for uniform mesh refinement.
        """
        for i in range(maxit):
            A, b = self.linear_system()
            A, b = self.apply_bc(A, b)
            self.xh[:] = self.solve(A, b)
            self.ph[:] = self.xh[:self.pspace.number_of_global_dofs()]
            self.uh[:] = self.xh[self.pspace.number_of_global_dofs():]
            ul2, pl2 = self.postprocess()
            self.show_p0(self.pde.solution)
            self.show_p0(self.uh)
            self.logger.info(f"{i}-th step with  u_L2 Error: {ul2},  p_L2 Error: {pl2}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.mesh.uniform_refine()


    @run.register("bisect")
    def run(self):
        pass
        
    @variantmethod("error")
    def postprocess(self):
        """
        Post-process the numerical solution to compute the error in L2 norm.
        """
        ul2 = self.mesh.error(self.pde.solution, self.uh)
        pl2 = self.mesh.error(self.pde.flux, self.ph)
        return ul2, pl2
    
    def show_p0(self,solution):
        """
        Visualize the primary variable solution in the Lagrange finite element space.

        This function interpolates the given solution onto the Lagrange space, computes the values
        at the node centers, and visualizes the result using a 3D surface plot.

        Parameters:
            solution (ndarray): The solution to be visualized, typically a scalar field defined in the Lagrange space.
        """
        u1 = self.uspace.interpolate(solution)
        node = self.mesh.entity('node') 
        cell = self.mesh.entity('cell') 
        num_nodes = len(node)
        node_values = bm.zeros(num_nodes)

        for i in range(len(cell)):
            for j in cell[i]:
                node_values[j] += u1[i]
                
        node_values /= bm.bincount(bm.concatenate(cell))
        from scipy.interpolate import griddata

        xi = bm.linspace(min(node[:, 0]), max(node[:, 0]))
        yi = bm.linspace(min(node[:, 1]), max(node[:, 1]))
        xi, yi = bm.meshgrid(xi, yi)

        zi = griddata((node[:, 0], node[:, 1]), node_values, (xi, yi), method='linear')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap='jet', linewidth=0, antialiased=False, edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('True Solution')
        ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        fig.colorbar(surf)
        plt.show()

    def show_rt(self, solution):
        """
        Visualize the Raviart-Thomas (RT) finite element solution on the mesh.

        This function interpolates the given solution onto the RT space, computes the values
        at the edge centers, and visualizes the result using a 3D contour plot.

        Parameters:
            solution (ndarray): The solution to be visualized, typically a vector field defined in the RT space.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        node_center = 0.5 * (node[edge[:, 0]] + node[edge[:, 1]])
        p_solution = self.pspace.interpolation(solution)
        x = node_center[:, 0]
        y = node_center[:, 1]
        surf = plt.tricontourf(x, y, p_solution, levels=50, cmap='jet', edgecolor='none')
        ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        fig.colorbar(surf)  
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Exact Solution')
        plt.show()
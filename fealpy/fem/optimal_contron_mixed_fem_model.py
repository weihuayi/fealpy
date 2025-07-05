from typing import Optional,Union
from fealpy.model import ComputationalModel
from fealpy.model.optimal_control import OPCPDEDataT

from fealpy.functionspace import RaviartThomasFESpace2d
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.fem import DivIntegrator
from fealpy.fem import OPCSIntegrator,OPCIntegrator
from fealpy.fem import BlockForm
from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve
from fealpy.model import PDEDataManager
from fealpy.decorator import variantmethod
from fealpy.mesh import Mesh


class OPCMixedFEMModel(ComputationalModel):
    """
    OPCRTFEMModel: Optimal Control Problem Raviart-Thomas Finite Element Model

    This class implements a 2D optimal control PDE solver using the Raviart-Thomas finite element method (RT FEM).
    It is designed for optimal control problems that require accurate flux approximation and state-control coupling.
    The model supports custom optimal control PDE data, mesh handling, linear system assembly, boundary condition application,
    solution routines, and error analysis.

    Parameters
    mesh : TriangleMesh
        The 2D triangle mesh object used for finite element discretization.
    c : float
        The regularization parameter or control parameter for the optimal control problem.
    pde : OPCPDEDataT, optional, default=None
        The optimal control PDE data object, including coefficients, source terms, and boundary conditions.
        If None, a built-in example problem is used.

    Attributes
    pde : OPCPDEDataT
        The current optimal control PDE data object, including coefficients, source terms, and boundary conditions.
    mesh : TriangleMesh
        The current 2D triangle mesh object used for finite element discretization.

    Methods
    run()
        Execute the FEM solution process and return the numerical solutions of the state and control variables.
    space()
        Return the finite element spaces used in the model.
    linear_system()
        Assemble the linear system (stiffness matrix and right-hand side) for the optimal control RT FEM.
    boundary_apply()
        Apply boundary conditions to the linear system.
    solve()
        Solve the linear system and return the FEM solutions of the state and control variables.
    show_mesh()
        Visualize the current mesh structure.
    L2_error()
        Compute the L2 error between the numerical and exact solutions.
    max_error()
        Compute the maximum error between the numerical and exact solutions.

    Notes
    This class uses a mixed finite element method (Raviart-Thomas space and piecewise constant space),
    suitable for optimal control problems with flux continuity and state-control coupling.
    It supports automatic boundary condition handling and error evaluation for algorithm verification and numerical experiments.

    Examples
    >>> model = OPCRTFEMModel(mesh, c)
    >>> p, u = model.run()
    >>> error_y, error_p = model.L2_error(y, p, exact_y, exact_p)
    >>> model.show_mesh()
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], options['uniform_refine'])
        self.set_order(options['space_degree'])
        self.solve.set(options['solve']) 
        
    def set_pde(self, pde: Union[OPCPDEDataT, str]="opc"):
        """
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('optimal_control').get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tri", n: int = 0, **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
        self.mesh.uniform_refine(n=n)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_order(self, p: int = 0):    
        self.p = p    
        
    def space(self, p: Optional[int] = None):
        """
        Set the finite element spaces for the model.
        
        Parameters
        p : int, optional
            The polynomial degree of the Raviart-Thomas space. If None, use the default degree.
        """
        if p is None:
            p = self.p
        self.pspace = RaviartThomasFESpace2d(self.mesh, p=p)
        self.uspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        return self.uspace, self.pspace
    
    def linear_system(self, mesh, p, s1, s2, s3, s4):
        """
        """
        self.pspace = RaviartThomasFESpace2d(mesh, p=p)
        self.uspace= LagrangeFESpace(mesh, p=p, ctype='D')  

        uLDOF = self.uspace.number_of_local_dofs()
        uGDOF = self.uspace.number_of_global_dofs()
        pLDOF = self.pspace.number_of_local_dofs()
        pGDOF = self.pspace.number_of_global_dofs()
        self.logger.info(f"Raviart-Thomas space: {self.pspace}, LDOF: {pLDOF}, GDOF: {pGDOF}")
        self.logger.info(f"Lagrange space: {self.uspace}, LDOF: {uLDOF}, GDOF: {uGDOF}")
        self.uh = self.uspace.function()
        self.ph = self.pspace.function()
        self.xh = bm.zeros((pGDOF + uGDOF,), dtype=bm.float64)
        
        bform1 = BilinearForm(self.pspace)
        bform1.add_integrator(OPCIntegrator(coef=self.pde.A_inverse, q=3))


        bform2 = BilinearForm((self.uspace,self.pspace))
        bform2.add_integrator(DivIntegrator(coef=-1, q=3))

        bform3 = BilinearForm((self.uspace,self.pspace))
        bform3.add_integrator(DivIntegrator(coef=1, q=3))

        bform4 = BilinearForm(self.uspace)
        bform4.add_integrator(ScalarMassIntegrator(coef=self.pde.c, q=3))

        M = BlockForm([[bform1,bform2],
                       [bform3.T,bform4]])
        A = M.assembly()
        
        lform1 = LinearForm(self.pspace)
        lform1.add_integrator(OPCSIntegrator(source=s1))
        lform1.add_integrator(OPCSIntegrator(source=s2))
        lform2 = LinearForm(self.uspace)
        lform2.add_integrator(ScalarSourceIntegrator(source=s3))
        lform2.add_integrator(ScalarSourceIntegrator(source=s4))
        F = lform2.assembly()
        G = lform1.assembly()
        b = bm.concatenate([G,F],axis=0)
        
        return A, b
    
    def apply_bc(self, A, b, gd):
        """
        Apply the boundary conditions to the linear system.
        """
        uspace, pspace = self.space()
        ugdof = uspace.number_of_global_dofs()
        G_apply = pspace.set_neumann_bc(gd)
        F = bm.zeros(ugdof, dtype=bm.float64)
        b_apply = bm.concatenate([G_apply,F],axis=0)
        b = b - b_apply
        return A, b
    
    @variantmethod("direct")
    def solve(self, A, b):
        from fealpy.solver import spsolve
        self.xh[:] = spsolve(A, b, solver='scipy')
        return self.xh
    
    @solve.register('amg')
    def solve(self, A, F):
        pass

    @solve.register('cg')
    def solve(self, A, b):
        from fealpy.solver import cg 
        self.xh[:], info = cg(A, b, maxit=5000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")

        
    @variantmethod("error")
    def postprocess(self, uh, ph, solution1, solution2):
        """
        Post-process the numerical solution to compute the error in L2 norm.
        """
        ul2 = self.mesh.error(solution1, uh)
        pl2 = self.mesh.error(solution2, ph)
        return ul2, pl2
  
    






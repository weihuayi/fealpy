from typing import Optional, Union
from ..model import ComputationalModel
from ..model.elliptic_mixed import EllipticPDEDataT

from ..functionspace import RaviartThomasFESpace2d
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarMassIntegrator, ScalarSourceIntegrator
from ..fem import DivIntegrator
from ..fem import BlockForm
from ..backend import backend_manager as bm
from ..fem import DirichletBC
from ..model import PDEDataManager
from ..decorator import variantmethod


class EllipticMixedFEMModel(ComputationalModel):
    """
    EllipticMixedFEMModel: Elliptic Mixed Finite Element Model

    This class implements a 2D elliptic PDE solver based on mixed finite element methods.
    It is suitable for elliptic problems requiring accurate flux approximation. The model supports custom PDE data,
    automatic mesh generation, linear system assembly, boundary condition handling, solution, and error analysis.

    Parameters
    pde : EllipticPDEDataT, optional, default=None
        The elliptic PDE data object, including coefficients, source terms, and boundary conditions.
        If None, a built-in example problem ('coscos') is used.

    Attributes
    pde : EllipticPDEDataT
        The current elliptic PDE data object, including coefficients, source terms, and boundary conditions.
    mesh : TriangleMesh
        The current 2D triangle mesh object used for finite element discretization.

    Methods
    run(maxit=4)
        Execute the complete FEM solution process and return the numerical solutions of the primary and auxiliary variables.
    set_init_mesh()
        Initialize and generate the 2D triangle mesh.
    linear_system()
        Assemble the linear system (stiffness matrix and right-hand side) for the elliptic mixed FEM.
    apply_bc()
        Apply boundary conditions to the linear system.
    solve()
        Solve the linear system and return the FEM solutions of the primary and auxiliary variables.
    postprocess()
        Compute the error between the numerical and exact solutions.

    Notes
    This class uses a mixed finite element method (Raviart-Thomas space and Lagrange space),
    suitable for elliptic problems with flux continuity requirements.
    It supports automatic boundary condition handling and error evaluation for algorithm verification and numerical experiments.

    Examples
    >>> model = EllipticMixedFEMModel()
    >>> model.run()
    >>> error_u, error_p = model.postprocess()
    """

    def __init__(self):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEDataManager("elliptic_mixed")
        
    def set_pde(self, pde: Union[EllipticPDEDataT, str]="coscos"):
        """
        Set the PDE data for the model.
        """
        if isinstance(pde, str):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

            
    @variantmethod("tri")
    def set_init_mesh(self, meshtype: str = "tri", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")
        
    @set_init_mesh.register("dis")
    def set_init_mesh(self,meshtype: str = "dis", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)
        
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")
        
        
    def set_space_degree(self, p: int = 1):    
        self.p = p
        
    @variantmethod("rt")
    def space(self, p: int = 1):
        """
        Set the finite element spaces for the model.
        
        Parameters
        p : int
            The polynomial degree of the finite element spaces.
        """
        self.p = p
        self.pspace = RaviartThomasFESpace2d(self.mesh, p=p)
        self.uspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        
        return self.uspace, self.pspace
        
    @space.register("bdm")
    def space(self, p: int = 1):
        """
        Set the BDM finite element spaces for the model.
        
        Parameters
        p : int
            The polynomial degree of the BDM finite element spaces.
        """
        from ..functionspace import BrezziDouglasMariniFESpace
        self.p = p
        self.pspace = BrezziDouglasMariniFESpace(self.mesh, p=p)
        self.uspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        
        return self.uspace, self.pspace
        
    def linear_system(self, mesh, p):
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
        """

        self.uspace, self.pspace = self.space(self.p)
        uLDOF = self.uspace.number_of_local_dofs()
        uGDOF = self.uspace.number_of_global_dofs()
        pLDOF = self.pspace.number_of_local_dofs()
        pGDOF = self.pspace.number_of_global_dofs()
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
        Apply the boundary conditions to the linear system.
        """
        uspace, pspace = self.space()
        ugdof = uspace.number_of_global_dofs()
        uGdof = uspace.number_of_global_dofs()
        pGdof = pspace.number_of_global_dofs()
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
        """
        A, b = self.linear_system(self.mesh, self.p)
        A, b = self.apply_bc(A, b)
        self.xh[:] = self.solve(A, b)
        self.ph[:] = self.xh[:self.pspace.number_of_global_dofs()]
        self.uh[:] = self.xh[self.pspace.number_of_global_dofs():]
        ul2, pl2 = self.postprocess()
        self.logger.info(f"u_L2 Error: {ul2},  p_L2 Error: {pl2}.")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        for i in range(maxit):
            A, b = self.linear_system(self.mesh, self.p)
            A, b = self.apply_bc(A, b)
            self.xh[:] = self.solve(A, b)
            self.ph[:] = self.xh[:self.pspace.number_of_global_dofs()]
            self.uh[:] = self.xh[self.pspace.number_of_global_dofs():]
            ul2, pl2 = self.postprocess()
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
from typing import Optional
from ..model import ComputationalModel, mregister
from ..model.optimal_control import OPCPDEDataT, get_example
from ..mesh import TriangleMesh

from ..functionspace import RaviartThomasFESpace2d
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarMassIntegrator, ScalarSourceIntegrator
from ..fem import DivIntegrator,OPCIntegrator,OPCSIntegrator
from ..fem import BlockForm
from ..backend import backend_manager as bm
from ..solver import spsolve
from ..fem import DirichletBC


class OPCRTFEMModel(ComputationalModel):
    """
    OPCRTFEMModel: Optimal Control Problem Raviart-Thomas Finite Element Model

    This class implements a 2D optimal control PDE solver using the Raviart-Thomas finite element method (RT FEM).
    It is designed for optimal control problems that require accurate flux approximation and state-control coupling.
    The model supports custom optimal control PDE data, mesh handling, linear system assembly, boundary condition application,
    solution routines, and error analysis.

    Parameters
    ----------
    mesh : TriangleMesh
        The 2D triangle mesh object used for finite element discretization.
    c : float
        The regularization parameter or control parameter for the optimal control problem.
    pde : OPCPDEDataT, optional, default=None
        The optimal control PDE data object, including coefficients, source terms, and boundary conditions.
        If None, a built-in example problem is used.

    Attributes
    ----------
    pde : OPCPDEDataT
        The current optimal control PDE data object, including coefficients, source terms, and boundary conditions.
    mesh : TriangleMesh
        The current 2D triangle mesh object used for finite element discretization.

    Methods
    -------
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
    -----
    This class uses a mixed finite element method (Raviart-Thomas space and piecewise constant space),
    suitable for optimal control problems with flux continuity and state-control coupling.
    It supports automatic boundary condition handling and error evaluation for algorithm verification and numerical experiments.

    Examples
    --------
    >>> model = OPCRTFEMModel(mesh, c)
    >>> p, u = model.run()
    >>> error_y, error_p = model.L2_error(y, p, exact_y, exact_p)
    >>> model.show_mesh()
    """

    def __init__(self,mesh, c, pde: Optional[OPCPDEDataT] = None):
        if pde is None:
            pde = get_example('opc')(c=c)
        self.pde = pde
        self.mesh = mesh

    def space(self):
        """
        Return the finite element spaces used in the model.
        """
        mesh = self.mesh
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D')
        return uspace,pspace


    def run(self):
        p, u = self.solve()
        return p, u


    def linear_system(self, source1=None, source2=None, source3=None, source4=None):
        """
        Assemble the linear system for the elliptic RT FEM model.
        """
        pde = self.pde
        mesh = self.mesh
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D') # discontinuous space
        ph = pspace.function()
        uh = uspace.function()

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        pgdof = pspace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        
        bform1 = BilinearForm(pspace)
        bform1.add_integrator(OPCIntegrator(coef=pde.A_inverse, q=3))

        bform2 = BilinearForm((uspace,pspace))
        bform2.add_integrator(DivIntegrator(coef=-1, q=3))
    
        bform3 = BilinearForm((uspace,pspace))
        bform3.add_integrator(DivIntegrator(coef=1, q=3))
    
        bform4 = BilinearForm(uspace)
        bform4.add_integrator(ScalarMassIntegrator(coef=pde.c, q=3))

        M = BlockForm([[bform1,bform2],
                       [bform3.T,bform4]])
        A = M.assembly()
        
        # Assemble the right-hand side
        lform1 = LinearForm(pspace)
        lform1.add_integrator(OPCSIntegrator(source=source1))
        lform1.add_integrator(OPCSIntegrator(source=source2))
        lform2 = LinearForm(uspace)
        lform2.add_integrator(ScalarSourceIntegrator(source=source3))
        lform2.add_integrator(ScalarSourceIntegrator(source=source4))
        F = lform2.assembly()
        G = lform1.assembly()
        b = bm.concatenate([G,F],axis=0)
        return A, b

    def boundary_apply(self, A, b, gd=None):
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
    
    def solve(self,A, b):
        """
        Solve the linear system.
        """
        uspace, pspace = self.space()
        pgdof = pspace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        p = pspace.function()
        u = uspace.function()
        x = spsolve(A, b,'scipy')
        p[:] = x[:pgdof]
        u[:] = x[pgdof:pgdof+ugdof]
        return p, u

    def show_mesh(self):

        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        self.mesh.add_plot(axes)
        plt.show()

    def L2_error(self, y, p, solution1=None,solution2=None):
        """
        Calculate the error of the solution.
        """
        mesh = self.mesh
        error_y = mesh.error(y, solution1)
        error_p = mesh.error(p, solution2)
        return error_y, error_p

    def max_error(self, y, p, solution1=None, solution2=None):
        """
        Calculate the maximum error of the solution.
        """
        mesh = self.mesh
        space1, space2 = self.space()
        yso = space1.interpolate(solution1)
        pso = space2.interpolate(solution2)
        max_y_error = bm.max(bm.abs(y - yso))
        max_p_error = bm.max(bm.abs(p - pso))
        return max_y_error, max_p_error







from typing import Optional
from ..model import ComputationalModel, mregister
from ..model.elliptic import EllipticPDEDataT, get_example
from ..mesh import TriangleMesh

from ..functionspace import RaviartThomasFESpace2d
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarMassIntegrator, ScalarSourceIntegrator,VectorMassIntegrator
from ..fem import DivIntegrator
from ..fem import BlockForm
from ..backend import backend_manager as bm
from ..solver import spsolve
from ..fem import DirichletBC

class EllipticRTFEMModel(ComputationalModel):
    """
    Class for the elliptic RT FEM model.
    """

    def __init__(self, pde: Optional[EllipticPDEDataT] = None):
        if pde is None:
            pde = get_example('coscos')()
        self.pde = pde



    def run(self, maxit=4):
        pde = self.pde 
        mesh = self.init_mesh()
        A,b = self.linear_system()
        A,b = self.boundary_apply()
        p,u = self.solve()
        return p, u

    def init_mesh(self):
        """
        Initialize the mesh for the elliptic RT FEM model.
        """
        domain = self.pde.domain()
        self.mesh = TriangleMesh.from_box(domain, nx=32, ny=32)
        return self.mesh

    def linear_system(self):
        """
        """
        pde = self.pde
        mesh = self.init_mesh()
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D') # discontinuous space
        ph = pspace.function()
        uh = uspace.function()

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        pgdof = pspace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        
        bform1 = BilinearForm(pspace)
        bform1.add_integrator(ScalarMassIntegrator(coef=pde.diffusion_coef_inv, q=3))
        #bform1.add_integrator(VectorMassIntegrator(coef=1, q=3))

        bform2 = BilinearForm((uspace,pspace))
        bform2.add_integrator(DivIntegrator(coef=-1, q=3))
    
        bform3 = BilinearForm((uspace,pspace))
        bform3.add_integrator(DivIntegrator(coef=1, q=3))
    
        bform4 = BilinearForm(uspace)
        bform4.add_integrator(ScalarMassIntegrator(coef=2, q=3))

        M = BlockForm([[bform1,bform2],
                       [bform3.T,bform4]])
        A = M.assembly()
        
         # 组装右端
        lform = LinearForm(uspace)
        lform.add_integrator(ScalarSourceIntegrator(source=pde.source))
        F = lform.assembly()
        G = bm.zeros(pgdof)
        b = bm.concatenate([G,F],axis=0)
        return A, b

    def boundary_apply(self):
        """
        Apply the boundary conditions to the linear system.
        """
        A, b = self.linear_system()
        # 组装边界条件 
        pde = self.pde
        mesh = self.init_mesh()
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D') # discontinuous space
        pgdof = pspace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        ispBdof = pspace.is_boundary_dof()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        isyBdof = bm.zeros(ugdof, dtype=bm.bool)
        isBdof = bm.concatenate([ispBdof,isyBdof],axis=0)
        fun = pspace.function()
        k,_ = pspace.set_dirichlet_bc(pde.grad_dirichlet,fun)
        k1 = bm.zeros(ugdof, dtype=bm.float64)
        k = bm.concatenate([k,k1],axis=0)
        bc = DirichletBC(space=(pspace,uspace),gd=k, threshold=isBdof)
        A, b = bc.apply(A, b)
        
        return A, b
    
    def solve(self):
        """
        Solve the linear system.
        """
        mesh = self.init_mesh()
        pde = self.pde
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D') # discontinuous space
        pgdof = pspace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        p = pspace.function()
        u = uspace.function()
        A, b = self.linear_system()
        A, b = self.boundary_apply()
        '''
        # 添加约束条件
        lform1 = LinearForm(uspace)
        lform1.add_integrator(ScalarSourceIntegrator(pde.source1))
        A3 = lform1.assembly()
        o = bm.zeros(1, dtype=bm.float64)
        G = bm.zeros(pgdof, dtype=bm.float64)
        A1 = bm.concatenate([G,A3],axis=0)
        pp = A1.shape[0]
        A2 = A1.reshape(pp,1)
        A1 = bm.concatenate([A1,o],axis=0).reshape(1,-1)
        M1  = A.toarray()
        M1 = bm.concatenate([M1,A2],axis=1)
        M1 = bm.concatenate([M1,A1],axis=0)   
        from scipy import sparse as sp
        A = sp.coo_matrix(M1).tocsr()
        b = bm.concatenate([b,o],axis=0)
        from scipy.sparse.linalg import spsolve
        '''
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
        
    def error(self):
        """
        Calculate the error of the solution.
        """
        p,u = self.solve()
        mesh = self.init_mesh()
        pde = self.pde
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D')
        p_solution = pspace.interpolate(pde.flux)
        error_p= mesh.error(p, p_solution)
        u_solution = uspace.interpolate(pde.solution)
        error_u = mesh.error(u, u_solution)
        return error_p, error_u








from fealpy.backend import backend_manager as bm                                   
from fealpy.vem import BilinearForm                              
from fealpy.vem import LinearForm                                  
from fealpy.vem import ScalarDiffusionIntegrator
from fealpy.vem import ScalarSourceIntegrator         
from fealpy.vem import DirichletBC                                
from fealpy.pde.poisson_2d import CosCosData                                       
from fealpy.mesh import TriangleMesh, PolygonMesh                                  
from fealpy.functionspace import ConformingScalarVESpace2d                         
from fealpy.solver import spsolve 

class PoissonCVEMModel:
    def __init__(self, pde, mesh, p):
        """
        virtual element method for Poisson equation in 2D.

        Parameters:
            pde: Poisson 2d. 

            mesh: PolygonMesh.

            p : degree of virtual element space.
        """
        bm.set_backend('numpy')
        self.pde = pde
        space = ConformingScalarVESpace2d(mesh, p=p)
        self.space = space
        self.assemble()

    def assemble(self):
        space = self.space
        pde = self.pde
        bform = BilinearForm(space)
        integrator = ScalarDiffusionIntegrator(coef=1, q=p+3)
        bform.add_integrator(integrator)
        A = bform.assembly()
        lform = LinearForm(space)
        integrator = ScalarSourceIntegrator(pde.source, q=p+3)
        lform.add_integrator(integrator)
        F = lform.assembly()
        bc1 = DirichletBC(space, pde.dirichlet)
        self.A, self.F = bc1.apply(A, F)

    def solver(self):
        self.uh = self.space.function()
        self.uh[:] = spsolve(self.A, self.F, "scipy")

    def error(self):
        pde = self.pde
        PI1 = self.space.PI1
        sh = self.space.project_to_smspace(self.uh)
        error = bm.zeros(2)
        error[0] = mesh.error(sh.value, pde.solution)
        error[1] = mesh.error(sh.grad_value, pde.gradient)
        return error

    def run(self):
        model.solver()
        error = model.error()
        print('L2 error:', error[0])
        print('H1 error:', error[1])

if __name__ == '__main__':
    pde = CosCosData()
    p = 2
    n = 4
    mesh = PolygonMesh.from_box([0,1,0,1],n,n,device='cpu')
    model = PoissonCVEMModel(pde, mesh, p)
    model.run()



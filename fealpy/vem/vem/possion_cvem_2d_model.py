from fealpy.backend import backend_manager as bm                                   
from fealpy.vem.vem.bilinear_form import BilinearForm                              
from fealpy.vem.vem.linear_form import LinearForm                                  
from fealpy.vem.vem.scalar_diffusion_integrator import ScalarDiffusionIntegrator
from fealpy.vem.vem.scalar_source_integrator import ScalarSourceIntegrator         
from fealpy.vem.vem.dirichlet_bc import DirichletBC                                
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
if __name__ == "__main__":
    n = 4
    p = 5
    errorMatrix = bm.zeros((2, 4))
    for i in range(4):
        mesh = PolygonMesh.from_box([0,1,0,1],n, n, device='cpu')
        pde = CosCosData()
        model = PoissonCVEMModel(pde, mesh, p=p)
        model.solver()
        errorMatrix[:, i] = model.error()
        n = n*2
    print('errorMatrix',errorMatrix)
    print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
    print("order : ", bm.log2(errorMatrix[1,:-1]/errorMatrix[1,1:]))



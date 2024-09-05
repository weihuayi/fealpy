from typing import Optional

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.decorator import barycentric, cartesian

from fealpy.experimental.fem import BilinearForm, LinearForm
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem import LinearElasticIntegrator
from fealpy.experimental.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.experimental.fem import ScalarSourceIntegrator

from fealpy.experimental.fem import DirichletBC

from app.fracturex.fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from app.fracturex.fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory

from scipy.sparse.linalg import lgmres, cg, spsolve


class MainSolver:
    def __init__(self, mesh, material_params, p=1, q=None, method='HybridModel'):
        """
        Parameters
        ----------
        material_params : material properties 'lam' : lame params, 'mu' : shear modulus , 'E' : Young's modulus, 'nu' : Poisson's ratio, 'Gc' : critical energy release rate, 'l0' : length scale
        method : str
            The method of stress decomposition method
        """
        self.mesh = mesh
        self.p = p
        self.q = self.p+2 if q is None else q

        self.EDFunc = EDFunc()
        self.pfcm = PhaseFractureMaterialFactory.create('HybridModel', material_params, self.EDFunc)
        
        self.set_lfe_space()
        self.Gc = material_params['Gc']
        self.l0 = material_params['l0']

    def solve(self):
        """
        Solve the phase field fracture problem.
        """

        
        pass

    def newton_raphson(self):
        """
        Newton-Raphson iteration
        """
        pass

    def load_displacement(self, disp_increment_boundary):
        """
        Set the displacement increment to be applied at each time step
        """
        index = disp_increment_boundary
        self.uh[index] = 1
        self.pfcm.update_disp(self.uh)

    def solve_displacement(self):
        """
        Solve the displacement field.
        """
        uh = self.uh
        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q))
        A = ubform.assembly()
        R = A@uh[:]
        
 #       self.force = np.sum(-R[force_ubd.flatten()])

#        ubc = DirichletBC(self.tspace, 0, fixed_ubd)

#        A, R = ubc.apply(A, R)
#        du.flat[:] = lgmres(A, R, tol=1e-14)[0]
#        uh.flat[:] += du.flat[:]
#        self.pfcm.update_disp(uh)
        

    def solve_phase_field(self):
        """
        Solve the phase field.
        """
    
        @barycentric
        def coef(bc, index):
            return 2*self.pfcm.maximum_historical_field(bc)
        
        Gc = self.Gc
        l0 = self.l0
        d = self.d

        dbfrom = BilinearForm(self.space)
        dbfrom.add_integrator(ScalarDiffusionIntegrator(Gc*l0, q=self.q))
        dbfrom.add_integrator(ScalarMassIntegrator(Gc/l0, q=self.q))
        dbfrom.add_integrator(ScalarMassIntegrator(coef, q=self.q))
        A = dbfrom.assembly()

        dlfrom = LinearForm(self.space)
        dlfrom.add_integrator(ScalarSourceIntegrator(coef, q=self.q))
        R = dlfrom.assembly()
        R -= A@d[:]


    def set_lfe_space(self):
        """
        Set the function space for the problem.
        """
        self.space = LagrangeFESpace(self.mesh, self.p)
        self.tspace = TensorFunctionSpace(self.space, (self.mesh.geo_dimension(), -1))

        self.d = self.space.function()
        self.uh = self.tspace.function()

        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)
        
    
    def vtkfile(self):
        pass


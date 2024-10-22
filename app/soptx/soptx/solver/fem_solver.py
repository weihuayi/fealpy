from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem import DirichletBC

from fealpy.sparse import CSRTensor

from fealpy.solver import cg, spsolve

from app.soptx.soptx.utils.timer import timer


class FEMSolver:
    def __init__(self, material_properties, tensor_space: TensorFunctionSpace, pde):
        """
        Initialize the FEMSolver with the provided parameters.
        """
        self.material_properties = material_properties
        self.tensor_space = tensor_space
        self.pde = pde

    def assemble_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix using the material properties and integrator.
        """
        integrator = LinearElasticIntegrator(material=self.material_properties, 
                                            q=self.tensor_space.p+3)
        # KE = integrator.assembly(space=self.tensor_space)
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly(format='csr')

        return K
    
    def assemble_force_vector(self):
        """
        Assemble the global force vector using the force function.
        """
        force = self.pde.force
        F = self.tensor_space.interpolate(force)

        return F

    def apply_boundary_conditions(self, K: CSRTensor, F: TensorLike) -> TensorLike:
        """
        Apply boundary conditions to the stiffness matrix and force vector.
        """
        dirichlet = self.pde.dirichlet
        threshold = self.pde.threshold
 

        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(), 
                        dtype=bm.float64, device=bm.get_device(self.tensor_space))
        isBdDof = self.tensor_space.is_boundary_dof(threshold=threshold, method='interp')

        F = F - K.matmul(uh_bd)
        F[isBdDof] = uh_bd[isBdDof]

        dbc = DirichletBC(space=self.tensor_space)
        K = dbc.apply_matrix(matrix=K, check=True)

        return K, F

    def solve(self, solver_method) -> TensorLike:
        """
        Solve the displacement field.
        """

        # tmr = timer("FEM Solver")
        # next(tmr)
        tmr = None

        K0 = self.assemble_stiffness_matrix()
        if tmr:
            tmr.send('Assemble Stiffness Matrix')

        F0 = self.assemble_force_vector()
        
        K, F = self.apply_boundary_conditions(K=K0, F=F0)
        if tmr:
            tmr.send('Apply Boundary Conditions')
        
        uh = self.tensor_space.function()

        if solver_method == 'cg':
            uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
            if tmr:
                tmr.send('Solve System with CG')
        elif solver_method == 'spsolve':
            uh[:] = spsolve(K, F, solver='mumps')
            if tmr:
                tmr.send('Solve System with spsolve')
        else:
            raise ValueError(f"Unsupported solver method: {solver_method}")
        
        if tmr:
            tmr.send(None)

        return uh

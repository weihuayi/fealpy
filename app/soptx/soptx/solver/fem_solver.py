from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg

class FEMSolver:
    def __init__(self, material_properties, tensor_space, pde):
        """
        Initialize the FEMSolver with the provided parameters.

        Args:
            material_properties: MaterialProperties object defining material behavior.
            tensor_space: TensorFunctionSpace object for the computational space.
            pde: PDEData object defining boundary conditions and forces.
            solver_method (str): The method used to solve the system 
                (e.g., 'mumps' for direct or 'cg' for iterative).

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
        KE = integrator.assembly(space=self.tensor_space)
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly()

        return K

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions to the stiffness matrix and force vector.
        """
        force = self.pde.force
        dirichlet = self.pde.dirichlet
        is_dirichlet_boundary_edge = self.pde.is_dirichlet_boundary_edge
        is_dirichlet_node = self.pde.is_dirichlet_node
        is_dirichlet_direction = self.pde.is_dirichlet_direction
        
        K = self.assemble_stiffness_matrix()
        F = self.tensor_space.interpolate(force)

        dbc = DBC(space=self.tensor_space, gd=dirichlet, left=False)
        F = dbc.check_vector(F)

        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(), dtype=F.dtype)

        uh_bd, isDDof = self.tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_bd,
                                                    threshold=(is_dirichlet_boundary_edge, 
                                                               is_dirichlet_node, 
                                                               is_dirichlet_direction))

        F = F - K.matmul(uh_bd)
        F[isDDof] = uh_bd[isDDof]

        K = dbc.check_matrix(K)
        indices = K.indices()
        new_values = bm.copy(K.values())
        IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
        new_values[IDX] = 0

        K = COOTensor(indices, new_values, K.sparse_shape)
        index, = bm.nonzero(isDDof)
        one_values = bm.ones(len(index), **K.values_context())
        one_indices = bm.stack([index, index], axis=0)
        K1 = COOTensor(one_indices, one_values, K.sparse_shape)
        K = K.add(K1).coalesce()

        return K, F

    def solve(self, solver_method) -> TensorLike:
        """
        Solve the displacement field.

        Returns:
            TensorLike: The displacement vector.
        """

        K, F = self.apply_boundary_conditions()
        uh = self.tensor_space.function()
        
        if K is None or F is None:
            raise ValueError("Stiffness matrix K or force vector F has not been assembled.")
        if solver_method == 'cg':
            uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
        elif solver_method == 'mumps':
            raise NotImplementedError("Direct solver using MUMPS is not implemented.")
        else:
            raise ValueError(f"Unsupported solver method: {solver_method}")

        return uh

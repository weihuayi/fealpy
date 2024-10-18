from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem import DirichletBC as DBC
from fealpy.sparse import COOTensor
from fealpy.solver import cg

class FEMSolver:
    def __init__(self, material_properties, tensor_space, boundary_conditions):
        """
        Initialize the FEMSolver with the provided parameters.

        Args:
            material_properties: MaterialProperties object defining material behavior.
            tensor_space: TensorFunctionSpace object for the computational space.
            boundary_conditions: BoundaryConditions object defining boundary conditions.
        """
        self.material_properties = material_properties
        self.tensor_space = tensor_space
        self.boundary_conditions = boundary_conditions

        self.uh = tensor_space.function()
        self.KE = None
        self.K = None
        self.F = None
        
        self.solved = False 
        
        self.assemble_stiffness_matrix()
        self.apply_boundary_conditions()

    def assemble_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix using the material properties and integrator.
        """
        integrator = LinearElasticIntegrator(material=self.material_properties, q=self.tensor_space.p+3)
        self.KE = integrator.assembly(space=self.tensor_space)
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        self.K = bform.assembly()

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions to the stiffness matrix and force vector.
        """
        force = self.boundary_conditions.force
        dirichlet = self.boundary_conditions.dirichlet
        is_dirichlet_boundary_edge = self.boundary_conditions.is_dirichlet_boundary_edge
        is_dirichlet_node = self.boundary_conditions.is_dirichlet_node
        is_dirichlet_direction = self.boundary_conditions.is_dirichlet_direction

        self.F = self.tensor_space.interpolate(force)

        dbc = DBC(space=self.tensor_space, gd=dirichlet, left=False)
        self.F = dbc.check_vector(self.F)

        isDDof = self.tensor_space.is_boundary_dof(threshold=(
            is_dirichlet_boundary_edge, is_dirichlet_node, is_dirichlet_direction))

        self.uh = self.tensor_space.boundary_interpolate(gD=dirichlet, uh=self.uh,
                                                        threshold=is_dirichlet_boundary_edge)

        self.F = self.F - self.K.matmul(self.uh[:])
        self.F[isDDof] = self.uh[isDDof]

        self.K = dbc.check_matrix(self.K)
        indices = self.K.indices()
        new_values = bm.copy(self.K.values())
        IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
        new_values[IDX] = 0

        self.K = COOTensor(indices, new_values, self.K.sparse_shape)
        index, = bm.nonzero(isDDof)
        one_values = bm.ones(len(index), **self.K.values_context())
        one_indices = bm.stack([index, index], axis=0)
        K1 = COOTensor(one_indices, one_values, self.K.sparse_shape)
        self.K = self.K.add(K1).coalesce()

    def solve(self) -> TensorLike:
        """
        Solve the displacement field.

        Returns:
            TensorLike: The displacement vector.
        """
        if self.K is None or self.F is None:
            raise ValueError("Stiffness matrix K or force vector F has not been assembled.")
        self.uh[:] = cg(self.K, self.F, maxiter=5000, atol=1e-14, rtol=1e-14)

        return self.uh
    
    def get_element_stiffness_matrix(self) -> TensorLike:
        """
        Get the element stiffness matrix.

        Returns:
            TensorLike: The element stiffness matrix KE.
        """
        return self.KE
    
    def get_element_displacement(self) -> TensorLike:
        """
        Get the displacement vector for each element.

        Returns:
            TensorLike: The displacement vector for each element (uhe).
        """
        if not self.solved:
            self.solve()
        cell2ldof = self.tensor_space.cell_to_dof()
        uhe = self.uh[cell2ldof]

        return uhe
    
    def get_global_stiffness_matrix(self) -> TensorLike:
        """
        Get the global stiffness matrix.

        Returns:
            TensorLike: The global stiffness matrix K.
        """
        if self.K is None:
            self.assemble_stiffness_matrix()
        return self.K

    def get_global_source_vector(self) -> TensorLike:
        """
        Get the global force vector.

        Returns:
            TensorLike: The global force vector F.
        """
        if self.F is None:
            self.apply_boundary_conditions()
        return self.F

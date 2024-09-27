from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg

class FEMSolver:
    def __init__(self, material_properties, 
                tensor_space, pde, solver_method):
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
        self.solver_method = solver_method

        self.uh = tensor_space.function()
        # self.uh_bd = tensor_space.function()
        self.KE = None
        self.K = None
        self.F = None
        
        self.solved = False 

    def assemble_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix using the material properties and integrator.
        """
        integrator = LinearElasticIntegrator(material=self.material_properties, 
                                            q=self.tensor_space.p+3)
        self.KE = integrator.assembly(space=self.tensor_space)
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        self.K = bform.assembly()

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions to the stiffness matrix and force vector.
        """
        force = self.pde.force
        dirichlet = self.pde.dirichlet
        is_dirichlet_boundary_edge = self.pde.is_dirichlet_boundary_edge
        is_dirichlet_node = self.pde.is_dirichlet_node
        is_dirichlet_direction = self.pde.is_dirichlet_direction
        
        KFull1 = self.K.to_dense().round(4)
        self.F = self.tensor_space.interpolate(force)

        dbc = DBC(space=self.tensor_space, gd=dirichlet, left=False)
        self.F = dbc.check_vector(self.F)

        # isDDof = self.tensor_space.is_boundary_dof(threshold=(
        #     is_dirichlet_boundary_edge, is_dirichlet_node, is_dirichlet_direction))
        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(), dtype=self.F.dtype)
        uh_bd, isDDof = self.tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_bd,
                                                    threshold=(is_dirichlet_boundary_edge, 
                                                               is_dirichlet_node, 
                                                               is_dirichlet_direction))

        self.F = self.F - self.K.matmul(uh_bd)
        self.F[isDDof] = uh_bd[isDDof]

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
        self.K = None 
        self.F = None

        self.assemble_stiffness_matrix()
        self.apply_boundary_conditions()
        KFull2 = self.K.to_dense().round(4)
        F2 = self.F

        
        if self.K is None or self.F is None:
            raise ValueError("Stiffness matrix K or force vector F has not been assembled.")
        if self.solver_method == 'cg':
            self.uh[:] = cg(self.K, self.F, maxiter=5000, atol=1e-14, rtol=1e-14)
        elif self.solver_method == 'mumps':
            raise NotImplementedError("Direct solver using MUMPS is not implemented.")
        else:
            raise ValueError(f"Unsupported solver method: {self.solver_method}")

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
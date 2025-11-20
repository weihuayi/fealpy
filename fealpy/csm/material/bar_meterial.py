from typing import Optional, Tuple
from builtins import float, str
from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class BarMaterial(LinearElasticMaterial):
    """Material properties for 3D bars.
    Parameters:
        name (str): The name of the material.
        model (object): The model containing the bar's geometric and material properties.
        E (float): The elastic modulus of the material.
        mu (float): The shear modulus of the material.
    """
    def __init__(self, 
                name: str,
                model,
                elastic_modulus: Optional[float] = None,
                poisson_ratio: Optional[float] = None,
                shear_modulus: Optional[float] = None) -> None:
        
        super().__init__(name=name, 
                        elastic_modulus= elastic_modulus, 
                        poisson_ratio=poisson_ratio,
                        shear_modulus=shear_modulus)
        
        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.mu = self.get_property('shear_modulus')

        self.model = model 
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  [bar]  E           : {self.E}\n"
        s += f"  [bar]  nu          : {self.nu}\n"
        s += f"  [bar]  mu          : {self.mu}\n"
        s += ")"
        return s
    
    def linear_basis(self, x: float, l: float) -> TensorLike:
        """Linear shape functions for a bar material.
        Parameters:
            x (float): Local coordinate along the bar axis.
            l (float): Length of the bar element.
        Returns:
            b (TensorLike): Linear shape functions evaluated at xi.
        """
        xi = x / l  
        t = 1.0 / l
        b = bm.zeros((2, 2), dtype=bm.float64)
        b[0, 0] = 1 - xi
        b[0, 1] = xi
        b[1, 0] = -t
        b[1, 1] = t
        return b
    
    def stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for bar material."""
        E = self.E
        D = bm.array([[E, 0, 0],
                      [0, E, 0],
                      [0, 0, E]], dtype=bm.float64)
        return D
    
    def compute_strain_and_stress(self, 
                                   mesh,
                                   disp,
                                   ele_indices=None) -> Tuple[TensorLike, TensorLike]:
        """Calculate the strain and stress for bar elements.
            Compute axial strain: ε = (u1 - u0) / L
            Compute axial stress: σ = E * ε
        
        Parameters:
            mesh: The mesh containing element information.
            disp: The global displacement vector.
            ele_indices (Optional[int]): Indices of elements to process. If None, process all elements.

        Returns:
            Tuple[TensorLike, TensorLike]: Strain and stress vectors.
        """
        NC = mesh.number_of_cells()
        if ele_indices is None:
            ele_indices = range(NC)
            num_elements = NC
        else:
            num_elements = len(ele_indices)
            
        strain = bm.zeros((num_elements, 3), dtype=bm.float64)
        stress = bm.zeros((num_elements, 3), dtype=bm.float64)

        edge_lengths = mesh.edge_length()
        edge_tangents = mesh.edge_tangent()

        for idx, i in enumerate(ele_indices):
            cell = mesh.entity('cell', i)
            node0_idx, node1_idx = cell[0], cell[1]
            
            # Get element-specific tangent vector and length
            l = edge_lengths[i]
            tan = edge_tangents[i]
            unit_tan = tan / l

            u_node0 = disp[node0_idx]
            u_node1 = disp[node1_idx]
            
            # translational displacement 
            u0_trans = u_node0[:3]
            u1_trans = u_node1[:3]
            
            u0 = bm.dot(unit_tan, u0_trans)
            u1 = bm.dot(unit_tan, u1_trans)
            
            # Axial strain: (u1 - u0) / L
            axial_strain = (u1 - u0) / l

            strain[idx, 0] = axial_strain
            strain[idx, 1] = 0.0
            strain[idx, 2] = 0.0

            stress[idx, 0] = self.E * axial_strain
            stress[idx, 1] = 0.0
            stress[idx, 2] = 0.0

        return strain, stress
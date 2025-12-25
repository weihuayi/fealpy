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
                model: Optional[object] = None,
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
                               coord_transform=None,
                               ele_indices=None) -> Tuple[TensorLike, TensorLike]:
        """Calculate the strain and stress for bar elements.
            Compute axial strain: ε = (u1 - u0) / L
            Compute axial stress: σ = E * ε
        
        Parameters:
            mesh: The mesh containing element information.
            disp: The global displacement vector.
            coord_transform: Coordinate transformation matrix for each element.
            ele_indices (Optional[int]): Indices of elements to process. If None, process all elements.

        Returns:
            Tuple[TensorLike, TensorLike]: Strain and stress vectors.
        """
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        edge = mesh.entity('edge')
        l = mesh.entity_measure('cell')
        
        # Process specific elements or all
        if ele_indices is not None:
            edge = edge[ele_indices]
            l = l[ele_indices]
            num_elements = len(ele_indices)
        else:
            num_elements = NC
            
        strain = bm.zeros((num_elements, 3), dtype=bm.float64)
        stress = bm.zeros((num_elements, 3), dtype=bm.float64)
        
        u_edge = disp[edge]  # Shape: (num_elements, 2, GD)
        
        # [u1_x, u1_y, u1_z, u2_x, u2_y, u2_z]
        u_global = u_edge.reshape(num_elements, 2 * GD)
        
        if coord_transform is not None:
            u_local = bm.einsum('cij, cj -> ci', coord_transform,  u_global) # Shape: (num_elements, 2)
        else:
            u_local = u_edge

        # u_local[:, 0] is axial displacement of node 1 (local x-direction)
        # u_local[:, 1] is axial displacement of node 2 (local x-direction)
        # ε = [-1/L, 1/L] · [u0, u1] = (u1 - u0) / L
        axial_strain = (u_local[:, 1] - u_local[:, 0]) / l
        axial_stress = self.E * axial_strain
        
        strain[:, 0] = axial_strain
        stress[:, 0] = axial_stress

        return strain, stress

    def calculate_mises_stress(self, stress: TensorLike) -> TensorLike:
        """Calculate von Mises stress from the stress tensor.

        Parameters:
            stress (TensorLike): The stress tensor of shape (num_elements, 3).

        Returns:
            TensorLike: The von Mises stress of shape (num_elements,).
        """
        mstress = stress[:, 0]
        return bm.abs(mstress)
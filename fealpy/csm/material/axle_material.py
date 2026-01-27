from typing import Optional, Tuple
from builtins import float, str
from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm


class AxleMaterial():
    """Material properties for 3D axle elements.
    
    Parameters:
        name (str): The name of the material.
        model (object): The model containing the axle's geometric properties.
        k_axle (float): The axle stiffness coefficient (N/m).
        E (float, optional): Elastic modulus for stress calculation (Pa).
    """
    def __init__(self, 
                name: str,
                model,
                k_axle: float = 1.976e6,
                elastic_modulus: Optional[float] = None) -> None:
        
        self.model = model
        self.k_axle = k_axle
        self.E = elastic_modulus  # 仅用于应力计算

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.name}\n"
        s += f"  [axle]  k_axle    : {self.k_axle} N/m\n"
        if self.E is not None:
            s += f"  [axle]  E (for stress): {self.E} Pa\n"
        s += ")"
        return s
    
    
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
        """Calculate the strain and stress for axle elements.
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
        if self.E is None:
            raise ValueError("Elastic modulus (E) must be set for stress calculation")
        
        NC = mesh.number_of_cells()
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
        
        u_edge = disp[edge]  # Shape: (num_elements, 2, 2*GD)
        
        # [u1_x, u1_y, u1_z, u1_thetax, u1_thetay, u1_thetaz, u2_x, u2_y, u2_z,  u2_thetax, u2_thetay, u2_thetaz]
        u_global = u_edge.reshape(num_elements, 12)

        if coord_transform is not None:
            u_local = bm.einsum('cij, cj -> ci', coord_transform,  u_global) # Shape: (num_elements, 12)
            u_local_axial = u_local[:, [0, 6]]  # Shape: (num_elements, 2)
        else:
            u_local_axial = u_global[:, [0, 6]]  # Shape: (num_elements, 2)

        # u_local[:, 0] is axial displacement of node 1 (local x-direction)
        # u_local[:, 1] is axial displacement of node 2 (local x-direction)
        # ε = [-1/L, 1/L] · [u0, u1] = (u1 - u0) / L
        axial_strain = (u_local_axial[:, 1] - u_local_axial[:, 0]) / l
        axial_stress = self.E * axial_strain
        
        strain[:, 0] = axial_strain
        stress[:, 0] = axial_stress

        return strain, stress
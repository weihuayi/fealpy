from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike

# from ...soptx.cases.material_properties import MaterialProperties, SIMPInterpolation
from ...soptx.cases.geometry_properties import GeometryProperties
from ...soptx.cases.filter_properties import FilterProperties
from ...soptx.cases.constraint_conditions import ConstraintConditions
from ...soptx.cases.boundary_conditions import BoundaryConditions    
from ...soptx.cases.termination_criterias import TerminationCriteria

class MBBBeamCase:
    def __init__(self, case_name: str):
        """
        Initialize the MBB beam case with specified parameters.

        Args:
            case_name (str): Name of the specific case or reference paper.
        """
        self.case_name = case_name
        self.material_properties = None
        self.geometry_properties = None
        self.filter_properties = None
        self.constraint_conditions = None
        self.boundary_conditions = None
        self.termination_criterias = None
        self.initialize_case_parameters()

    def initialize_case_parameters(self):
        """
        Initialize parameters and variables based on the case name.
        """
        if self.case_name == "top88":
            self.geometry_properties = GeometryProperties(x_min=0.0, x_max=6.0, 
                                                          y_min=0.0, y_max=2.0)
            
            width, height = self.geometry_properties.get_dimensions()
            self.nx = int(width)
            self.ny = int(height)
            self.h = [width / self.nx, height / self.ny]

            self.constraint_conditions = ConstraintConditions()
            self.constraint_conditions.set_volume_constraint(is_on=True, vf=0.5)

            self.volfrac = self.constraint_conditions.get_constraints()['volume']['vf']
            self.rho = self.volfrac * bm.ones(self.nx * self.ny, dtype=bm.float64)

            self.material_properties = MaterialProperties(
                        E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
                        hypo="plane_stress", rho=self.rho,
                        interpolation_model=SIMPInterpolation())

            self.filter_properties = FilterProperties(
                                        nx=self.nx, ny=self.ny, 
                                        rmin=1.5, ft=1)

            def force(points: TensorLike) -> TensorLike:
    
                val = bm.zeros(points.shape, dtype=points.dtype)
                val[self.ny, 1] = -1
    
                return val
            
            def dirichlet(points: TensorLike) -> TensorLike:

                return bm.zeros(points.shape, dtype=points.dtype)
            
            def is_dirichlet_boundary_edge(edge_centers: TensorLike) -> TensorLike:

                left_edge = (edge_centers[:, 0] == 0.0)
                specific_edge = (edge_centers[:, 0] == self.nx) & (edge_centers[:, 1] == 0.5)
                
                result = left_edge | specific_edge

                return result
            
            def is_dirichlet_node() -> TensorLike:
                
                dirichlet_nodes = bm.zeros((self.nx+1)*(self.ny+1), dtype=bool)

                dirichlet_nodes[0:self.ny + 1] = True
                dirichlet_nodes[(self.ny + 1) * self.nx] = True

                return dirichlet_nodes
            
            def is_dirichlet_direction() -> TensorLike:
                
                direction_flags = bm.zeros(((self.nx + 1) * (self.ny + 1), 2), dtype=bool)

                direction_flags[0, 0] = True
                direction_flags[1, 0] = True
                direction_flags[2, 0] = True 
                direction_flags[(self.ny + 1) * self.nx, 1] = True
                # temp = bm.tensor([True, False])

                return direction_flags
            
            self.boundary_conditions = BoundaryConditions(
                force_func=force, 
                dirichlet_func=dirichlet,
                is_dirichlet_boundary_edge_func=is_dirichlet_boundary_edge,
                is_dirichlet_node_func=is_dirichlet_node,
                is_dirichlet_direction_func=is_dirichlet_direction
            )

            self.termination_criterias = TerminationCriteria(max_loop=2000, tol_change=0.01)

        else:
            raise ValueError(f"Case '{self.case_name}' is not defined.")

    def __repr__(self):
        """
        Return a string representation of the MBBBeamCase object.

        Returns:
            str: A string showing the case name and initialized parameters.
        """
        repr_str = (f"MBBBeamCase(\n"
                    f"  case_name = {self.case_name},\n"
                    f"  material_properties = {self.material_properties},\n"
                    f"  geometry_properties = {self.geometry_properties},\n"
                    f"  constraint_conditions = {self.constraint_conditions},\n"
                    f"  filter_properties = {self.filter_properties},\n"
                    f"  boundary_conditions = {self.boundary_conditions},\n"
                    f"  termination_criterias = {self.termination_criterias}\n")

        if hasattr(self, 'nx') and hasattr(self, 'ny'):
            repr_str += f"  nx = {self.nx}, ny = {self.ny},\n"

        if hasattr(self, 'rho'):
            repr_str += f"  initial_density = {self.rho},\n"

        if hasattr(self, 'h'):
            repr_str += f"  cell_sizes = {self.h}\n"

        repr_str += ")"
        return repr_str

# Example usage
if __name__ == "__main__":
    mbb_case = MBBBeamCase("top88")
    print(mbb_case)

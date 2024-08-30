from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike

from app.stopt.soptx.cases.material_properties import MaterialProperties
from app.stopt.soptx.cases.geometry_properties import GeometryProperties
from app.stopt.soptx.cases.filter_properties import FilterProperties
from app.stopt.soptx.cases.constraint_conditions import ConstraintConditions
from app.stopt.soptx.cases.boundary_conditions import BoundaryConditions    
from app.stopt.soptx.cases.termination_criterias import TerminationCriteria

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
            nx, ny = 60, 20
            rmin, ft = 1.5, 0

            self.material_properties = MaterialProperties(E0=1.0, Emin=1e-9, nu=0.3, penal=3.0)
            self.geometry_properties = GeometryProperties(x_min=0.0, x_max=float(nx), 
                                                        y_min=0.0, y_max=float(ny))
            self.filter_properties = FilterProperties(nx=nx, ny=ny, rmin=rmin, ft=ft)

            self.constraint_conditions = ConstraintConditions()
            self.constraint_conditions.set_volume_constraint(is_on=True, vf=0.5)

            def force(points: TensorLike) -> TensorLike:
    
                val = bm.zeros(points.shape, dtype=points.dtype)
                val[ny, 1] = -1
    
                return val
            
            def dirichlet(points: TensorLike) -> TensorLike:

                return bm.zeros(points.shape, dtype=points.dtype)
            
            def is_dirichlet_boundary_edge(points: TensorLike) -> TensorLike:

                temp = (points[:, 0] == 0.0)

                return temp
            
            def is_dirichlet_direction() -> TensorLike:

                temp = bm.tensor([True, False])

                return temp
            
            self.boundary_conditions = BoundaryConditions(
                force_func=force, 
                dirichlet_func=dirichlet,
                is_dirichlet_boundary_edge_func=is_dirichlet_boundary_edge,
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
        return (f"MBBBeamCase(\n"
            f"  case_name = {self.case_name},\n"
            f"  material_properties = {self.material_properties},\n"
            f"  geometry_properties = {self.geometry_properties},\n"
            f"  constraint_conditions = {self.constraint_conditions},\n"
            f"  filter_properties = {self.filter_properties},\n"
            f"  boundary_conditions = {self.boundary_conditions},\n"
            f"  termination_criterias = {self.termination_criterias}\n"
            f")")

# Example usage
if __name__ == "__main__":
    mbb_case = MBBBeamCase("top88")
    print(mbb_case)

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT

from builtins import int, str, float, object

from fealpy.experimental.typing import Union

from fealpy.experimental.opt.objective import Objective

from fealpy.experimental.mesh.mesh_base import Mesh

from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.functionspace.tensor_space import TensorFunctionSpace

from app.soptx.soptx.solver.fem_solver import FEMSolver

from app.soptx.soptx.utilfunc.calculate_ke0 import calculate_ke0

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties
from app.soptx.soptx.cases.boundary_conditions import BoundaryConditions

from app.soptx.soptx.filter.filter_properties import FilterProperties

from app.soptx.soptx.opt.volume_objective import VolumeConstraint

class ComplianceObjective(Objective):
    """
    Compliance Minimization Objective for Topology Optimization.

    This class implements the objective function (compliance) and Hessian matrix
    for compliance minimization in topology optimization.
    """
    
    def __init__(self,
                mesh: Mesh,
                space_degree: int,
                dof_per_node: int,
                dof_ordering: str,
                material_properties: ElasticMaterialProperties,
                filter_type: Union[int, str],
                filter_rmin: float,
                pde: object,
                solver_method: str,
                volume_constraint: VolumeConstraint) -> None:
        """
        Initialize the compliance objective function.
        """
        super().__init__()
        self.mesh = mesh
        self.space_degree = space_degree
        self.dof_per_node = dof_per_node
        self.dof_ordering = dof_ordering
        self.material_properties = material_properties
        self.filter_type = filter_type
        self.filter_rmin = filter_rmin
        self.pde = pde
        self.solver_method = solver_method
        self.volume_constraint = volume_constraint

        self.space = self._create_function_space(self.space_degree, 
                                                self.dof_per_node, self.dof_ordering)
        self.filter_properties = self._create_filter_properties(self.filter_type, 
                                                            self.filter_rmin)
        self.displacement_solver = self._create_displacement_solver()

        self.ke0 = calculate_ke0(material_properties=self.material_properties, 
                                tensor_space=self.space)

    def _create_function_space(self, degree: int, 
                            dof_per_node: int, dof_ordering: str) -> TensorFunctionSpace:
        """
        Create a TensorFunctionSpace instance based on the given degree, 
            DOF per node, and DOF ordering.

        Args:
            degree (int): Degree of the function space.
            dof_per_node (int): Number of degrees of freedom per node.
            dof_ordering (str): DOF ordering, either 'dof-priority' or 'gd-priority'.

        Returns:
            TensorFunctionSpace: An instance of TensorFunctionSpace.
        """
        space_C = LagrangeFESpace(mesh=self.mesh, p=degree, ctype='C')

        if dof_ordering == 'gd-priority':
            shape = (-1, dof_per_node)
        elif dof_ordering == 'dof-priority':
            shape = (dof_per_node, -1)
        else:
            raise ValueError("Invalid DOF ordering. Use 'gd-priority' or 'dof-priority'.")

        return TensorFunctionSpace(scalar_space=space_C, shape=shape)

    def _create_filter_properties(self, filter_type: Union[int, str], 
                                filter_rmin: float) -> FilterProperties:
        """
        Create a FilterProperties instance based on the given filter type and radius.

        Args:
            filter_type (Union[int, str]): Type of the filter 
                (either 'density', 'sensitivity', 0, or 1).
            rmin (float): Filter radius.

        Returns:
            FilterProperties: An instance of FilterProperties.
        """
        if filter_type == 'density' or filter_type == 0:
            ft = 0
        elif filter_type == 'sensitivity' or filter_type == 1:
            ft = 1
        else:
            raise ValueError("Invalid filter type. Use 'density', 'sensitivity', 0, or 1.")

        return FilterProperties(mesh=self.mesh, rmin=filter_rmin, ft=ft)

    def _create_displacement_solver(self) -> FEMSolver:
        """
        Create a FEMSolver instance based on the given solver method.

        Args:
            solver_method (str): The method used to solve the system (e.g., 'mumps' for direct or 'cg' for iterative).

        Returns:
            FEMSolver: An instance of FEMSolver with the specified solving method.
        """
        return FEMSolver(material_properties=self.material_properties,
                        tensor_space=self.space,
                        pde=self.pde)

    # def _compute_uhe_and_ce(self, rho: _DT):
    #     """
    #     Compute the element displacement and compliance energy for the given density.

    #     Parameters:
    #         rho (_DT): Design variable (density distribution).
        
    #     Returns:
    #         Tuple[_DT, _DT]: Element displacement (uhe) and compliance energy (ce).
    #     """
    #     ft = self.filter_properties.ft
    #     H = self.filter_properties.H
    #     Hs = self.filter_properties.Hs

    #     cell_measure = self.mesh.entity_measure('cell')

    #     if ft == 0:
    #         rho_phys = H.matmul(rho[:] * cell_measure) / H.matmul(cell_measure)
    #     elif ft == 1:
    #         rho_phys = rho
        
    #     material_properties = self.material_properties
    #     displacement_solver = self.displacement_solver
    #     ke0 = self.ke0

    #     material_properties.rho = rho_phys

    #     uh = displacement_solver.solve()
    #     cell2ldof = self.space.cell_to_dof()
    #     uhe = uh[cell2ldof]

    #     # uhe = displacement_solver.get_element_displacement()
    #     # E = material_properties.material_model()

    #     ce = bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe)

    #     return ce
    #     # return uhe, ce, E

    def fun(self, rho: _DT) -> float:
        """
        Compute the compliance based on the density.
        
        Parameters:
            rho (_DT): Design variable (density distribution).
        
        Returns:
            float: Compliance value.
        """
        # ft = self.filter_properties.ft
        # H = self.filter_properties.H
        # Hs = self.filter_properties.Hs

        # cell_measure = self.mesh.entity_measure('cell')

        # if ft == 0:
        #     rho_phys = H.matmul(rho[:] * cell_measure) / H.matmul(cell_measure)
        # elif ft == 1:
        #     rho_phys = rho
        
        material_properties = self.material_properties
        displacement_solver = self.displacement_solver
        ke0 = self.ke0

        material_properties.rho = rho

        uh = displacement_solver.solve(solver_method='cg')
        cell2ldof = self.space.cell_to_dof()
        uhe = uh[cell2ldof]

        ce = bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe)

        self.ce = ce

        E = self.material_properties.material_model()

        c = bm.einsum('c, c -> ', E, ce)
        
        return c
    
    def jac(self, rho: _DT) -> _DT:
        """
        Compute the gradient of compliance w.r.t. density.

        Parameters:
            rho (_DT): Design variable (density distribution).
        
        Returns:
            _DT: Gradient of the compliance.
        """
        material_properties = self.material_properties

        ce = self.ce

        dE = material_properties.material_model_derivative()
        dce = - bm.einsum('c, c -> c', dE, ce)

        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        cell_measure = self.mesh.entity_measure('cell')

        if ft == 0:
            # 先归一化再乘权重因子
            dce[:] = H.matmul(dce * cell_measure / H.matmul(cell_measure))
        elif ft == 1:
            rho_dce = bm.einsum('c, c -> c', rho[:], dce)
            filtered_dce = H.matmul(rho_dce)
            dce[:] = filtered_dce / Hs / bm.maximum(bm.array(0.001), rho[:])

        return dce


    def hess(self, rho: _DT, lambda_: dict) -> _DT:
        material_properties = self.material_properties

        material_properties.rho = rho
        E = material_properties.material_model()
        dE = material_properties.material_model_derivative()

        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        _, ce, _ = self._compute_uhe_and_ce(rho)
        coef = 2 * dE**2 / E
        hessf = coef.matmul(ce)

        if ft == 0:
            hessf = hessf
        elif ft == 1:
            hessf[:] = H.matmul(hessf) / Hs

        hessc = self._compute_constraint_hessian(rho)

        if 'ineq' in lambda_:
            hessian = bm.diag(hessf) + lambda_['ineq'] * hessc
        else:
            hessian = bm.diag(hessf)
        
        return hessian
    
    def _compute_constraint_hessian(self, rho: _DT) -> _DT:
        """
        Compute the Hessian of the constraint function for non-linear constraints.

        Parameters:
            rho (_DT): Current design variables (density distribution).

        Returns:
            _DT: Hessian matrix of the constraint function.
        """
        # For linear constraints, this would be zero
        # For non-linear constraints, provide the Hessian computation here
        return 0  # Placeholder for linear constraint case
    



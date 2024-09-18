from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT
from fealpy.experimental.opt.objective import Objective
from fealpy.experimental.mesh.mesh_base import Mesh
from fealpy.experimental.functionspace.tensor_space import TensorFunctionSpace

from app.soptx.soptx.utilfunc.fem_solver import FEMSolver
from app.soptx.soptx.utilfunc.calculate_ke0 import calculate_ke0
from app.soptx.soptx.cases.material_properties import MaterialProperties
from app.soptx.soptx.cases.filter_properties import FilterProperties


class ComplianceObjective(Objective):
    """
    Compliance Minimization Objective for Topology Optimization.

    This class implements the objective function (compliance) and Hessian matrix
    for compliance minimization in topology optimization.
    """
    
    def __init__(self,
                mesh: Mesh,
                space: TensorFunctionSpace,
                material_properties: MaterialProperties,
                filter_properties: FilterProperties,
                displacement_solver: FEMSolver) -> None:
        """
        Initialize the compliance objective function.
        """
        super().__init__()
        self.mesh = mesh
        self.space = space
        self.material_properties = material_properties
        self.filter_properties = filter_properties
        self.displacement_solver = displacement_solver

        self.ke0 = calculate_ke0(material_properties=self.material_properties, 
                                tensor_space=self.space)

    def _compute_uhe_and_ce(self, rho: _DT):
        """
        Compute the element displacement and compliance energy for the given density.

        Parameters:
            rho (_DT): Design variable (density distribution).
        
        Returns:
            Tuple[_DT, _DT]: Element displacement (uhe) and compliance energy (ce).
        """
        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        if ft == 0:
            rho_phys = rho
        elif ft == 1:
            rho_phys = H.matmul(rho) / Hs
        
        material_properties = self.material_properties
        displacement_solver = self.displacement_solver
        ke0 = self.ke0

        material_properties.rho = rho_phys

        uhe = displacement_solver.get_element_displacement()
        E = material_properties.material_model()

        ce = bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe)

        return uhe, ce, E

    def fun(self, rho: _DT) -> float:
        """
        Compute the compliance based on the density.
        
        Parameters:
            rho (_DT): Design variable (density distribution).
        
        Returns:
            float: Compliance value.
        """
        _, ce, E = self._compute_uhe_and_ce(rho)
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

        _, ce, _ = self._compute_uhe_and_ce(rho)

        dE = material_properties.material_model_derivative()
        dce = - bm.einsum('c, c -> c', dE, ce)

        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        if ft == 0:
            rho_dce = bm.einsum('c, c -> c', rho[:], dce)
            filtered_dce = H.matmul(rho_dce)
            dce[:] = filtered_dce / Hs / bm.maximum(bm.array(0.001), rho[:])
        elif ft == 1:
            dce[:] = H.matmul(dce) / Hs

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
    



from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT
from fealpy.experimental.opt.objective import Objective

from app.soptx.soptx.utilfunc.fem_solver import FEMSolver
from app.soptx.soptx.cases.material_properties import MaterialProperties
from app.soptx.soptx.cases.filter_properties import FilterProperties

class ComplianceObjective(Objective):
    """
    Compliance Minimization Objective for Topology Optimization.

    This class implements the objective function (compliance) and Hessian matrix
    for compliance minimization in topology optimization.
    """
    
    def __init__(self, KE0: _DT, 
                material_properties: MaterialProperties, 
                displacement_solver: FEMSolver, 
                filter_properties: FilterProperties) -> None:
        """
        Initialize the compliance objective function.
        
        Parameters:
            KE0 (_DT): Global element stiffness matrix.
            material_properties (MaterialProperties): Material properties object.
            displacement_solver (FEMSolver): Solver for displacement calculation.
            filter_properties (FilterProperties): Filter properties for optimization.
        """
        super().__init__()
        self.KE0 = KE0
        self.material_properties = material_properties
        self.displacement_solver = displacement_solver
        self.filter_properties = filter_properties

    def fun(self, rho: _DT) -> float:
        """
        Compute the compliance based on the density.
        
        Parameters:
            rho (_DT): Design variable (density distribution).
        
        Returns:
            Tuple[float, _DT]: Compliance value.
        """
        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        if ft == 0:
            rho_phys = rho
        elif ft == 1:
            rho_phys = H.matual(rho) / Hs
        
        material_properties = self.material_properties
        displacement_solver = self.displacement_solver
        KE0 = self.KE0

        material_properties.rho = rho_phys

        uhe = displacement_solver.get_element_displacement()
        E = material_properties.material_model()

        ce = bm.einsum('ci, cik, ck -> c', uhe, KE0, uhe)
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
        displacement_solver = self.displacement_solver
        KE0 = self.KE0

        material_properties.rho = rho

        uhe = displacement_solver.get_element_displacement()

        ce = bm.einsum('ci, cik, ck -> c', uhe, KE0, uhe)

        dE = material_properties.material_model_derivative()
        dce = - bm.einsum('c, c -> c', dE, ce)

        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        if ft == 0:
            rho_dce = bm.multiply(rho, dce)
            filtered_dce = H.matmul(rho_dce)
            dce[:] = filtered_dce / Hs / bm.maximum(bm.array(0.001), rho)
        elif ft == 1:
            dce[:] = H.matmul(dce) / Hs

        return dce

    def hessp(self, rho: _DT, lambda_: _DT) -> _DT:
        material_properties = self.material_properties
        displacement_solver = self.displacement_solver
        KE0 = self.KE0

        material_properties.rho = rho
        
        E = material_properties.material_model()
        dE = material_properties.material_model_derivative()
        uhe = displacement_solver.get_element_displacement()

        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        ce = bm.einsum('ci, cik, ck -> c', uhe, KE0, uhe)
        coef = 2 * dE**2 / E
        hessf = coef.matmul(ce)
        hessf[:] = H.matmul(hessf) / Hs

        hessc = 0
        hessian = bm.diag(hessf) + lambda_.ineqnonlin * hessc
        
        return hessian
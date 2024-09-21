from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT

from fealpy.experimental.opt.objective import Constraint
from fealpy.experimental.mesh.mesh_base import Mesh

from app.soptx.soptx.cases.filter_properties import FilterProperties


class VolumeConstraint(Constraint):
    def __init__(self, 
                mesh: Mesh,
                volfrac: float,
                filter_properties: FilterProperties) -> None:
        """
        Initialize the volume constraint for topology optimization.

        Parameters:
            mesh (Mesh): The mesh object containing information about the finite element mesh.
            volfrac (float): The desired volume fraction of the structure.
            filter_properties (FilterProperties): The filter properties containing H, Hs, 
                and the filter type (ft), which determines the use of density or sensitivity filter.
        """
        self.mesh = mesh
        self.volfrac = volfrac
        self.filter_properties = filter_properties

        super().__init__(self.fun, self.jac, type='ineq')

    def fun(self, rho: _DT) -> float:
        """
        Compute the volume constraint function.

        This function calculates the total volume of the material distribution 
        in the design domain and subtracts the desired volume fraction. The 
        result indicates whether the current design violates the volume constraint.

        Parameters:
            rho (_DT): Design variable (density distribution), representing the 
                material distribution in the design domain.
        
        Returns:
            float: The volume constraint value, where a positive value indicates 
                a feasible design, and a negative value indicates a violation of 
                the volume constraint.
        """
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs
        ft = self.filter_properties.ft

        cell_measure = self.mesh.entity_measure('cell')

        if ft == 0:
            rho_phys = rho
        elif ft == 1:
            rho_phys = H.matmul(rho[:] * cell_measure) / H.matmul(cell_measure)
            # rho_phys = H.matmul(rho) / Hs

        # 假设所以单元面积相等（均匀网格）
        # NC = self.mesh.number_of_cells()
        # cneq = bm.sum(rho_phys[:]) - self.volfrac * NC

        # 单元面积不等（非均匀网格）
        volfrac_true = bm.einsum('c, c -> ', cell_measure, rho_phys[:]) / bm.sum(cell_measure)
        gneq = volfrac_true - self.volfrac
        
        return gneq

    def jac(self, rho: _DT) -> _DT:
        """
        Compute the gradient of the volume constraint function.

        This function returns the gradient of the volume constraint with respect 
        to the design variables. The gradient is typically used in the optimization 
        process to update the design variables while satisfying the volume constraint.

        Returns:
            _DT: The gradient of the volume constraint, representing the sensitivity 
                of the constraint to changes in the design variables.
        """
        gradg = self.mesh.entity_measure('cell')

        return gradg

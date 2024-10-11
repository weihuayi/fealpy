from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT

from fealpy.experimental.typing import Union

from fealpy.experimental.opt.objective import Constraint
from fealpy.experimental.mesh.mesh_base import Mesh

from app.soptx.soptx.filter.filter_properties import FilterProperties


class VolumeConstraint(Constraint):
    def __init__(self, 
                 mesh: Mesh,
                 volfrac: float,
                 filter_type: Union[int, str],
                 filter_rmin: float) -> None:
        """
        Initialize the volume constraint for topology optimization.

        Parameters:
            mesh (Mesh): The mesh object containing information about the finite element mesh.
            volfrac (float): The desired volume fraction of the structure.
            filter_type (Union[int, str]): The filter type, either 'density', 'sensitivity', 0, or 1.
            filter_rmin (float): The filter radius, which controls the minimum feature size.
        """
        self.mesh = mesh
        self.volfrac = volfrac

        self.filter_properties = self._create_filter_properties(filter_type, filter_rmin)

        super().__init__(self.fun, self.jac, type='ineq')

    def _create_filter_properties(self, filter_type: Union[int, str], filter_rmin: float) -> FilterProperties:
        """
        Create a FilterProperties instance based on the given filter type and radius.

        Args:
            filter_type (Union[int, str]): Type of the filter (either 'density', 'sensitivity', 0, or 1).
            filter_rmin (float): Filter radius.

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

        NC = self.mesh.number_of_cells()
        cell_measure = self.mesh.entity_measure('cell')

        volfrac_true = bm.einsum('c, c -> ', cell_measure, rho[:]) / bm.sum(cell_measure)
        gneq = (volfrac_true - self.volfrac) * NC
        # gneq = bm.sum(rho_phys[:]) - self.volfrac * NC
        
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
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs
        ft = self.filter_properties.ft

        cell_measure = self.mesh.entity_measure('cell')
        dge = cell_measure.copy()

        if ft == 0:
            # 先归一化再乘权重因子
            dge[:] = H.matmul(dge * cell_measure / H.matmul(cell_measure))
        elif ft == 1:
            pass

        return dge

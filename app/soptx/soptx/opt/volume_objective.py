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
        """
        self.mesh = mesh
        self.volfrac = volfrac

        self.filter_properties = self._create_filter_properties(filter_type, filter_rmin)

        super().__init__(self.fun, self.jac, type='ineq')

    def _create_filter_properties(self, filter_type: Union[int, str], 
                                filter_rmin: float) -> Union[FilterProperties, None]:
        """
        Create a FilterProperties instance based on the given filter type and radius.
        """
        if filter_type == 'None':
            return None 
    
        if filter_type == 'density' or filter_type == 0:
            ft = 0
        elif filter_type == 'sensitivity' or filter_type == 1:
            ft = 1
        elif filter_type == 'heaviside' or filter_type == 2:
            ft = 2
        else:
            raise ValueError("Invalid filter type. \
                            Use 'density', 'sensitivity', 'heaviside', 0, 1, or 2.")

        return FilterProperties(mesh=self.mesh, rmin=filter_rmin, ft=ft)

    def fun(self, rho: _DT) -> float:
        """
        Compute the volume constraint function.
        """
        NC = self.mesh.number_of_cells()
        cell_measure = self.mesh.entity_measure('cell')

        volfrac_true = bm.einsum('c, c -> ', cell_measure, rho[:]) / bm.sum(cell_measure)
        gneq = (volfrac_true - self.volfrac) * NC
        # gneq = bm.sum(rho[:]) - self.volfrac * NC
        
        return gneq

    def jac(self, rho: _DT, beta: float = None, rho_tilde: _DT = None) -> _DT:
        """
        Compute the gradient of the volume constraint function.
        """
        cell_measure = self.mesh.entity_measure('cell')
        dge = bm.copy(cell_measure)

        if self.filter_properties is None:
            return dge
        
        H = self.filter_properties.H
        ft = self.filter_properties.ft

        if ft == 0:
            # first normalize, then apply weight factor
            dge[:] = H.matmul(dge * cell_measure / H.matmul(cell_measure))
        elif ft == 1:
            print("Notice: Volume constraint sensitivity is not filtered when using sensitivity filter (ft == 1).")
        elif ft == 2:
            if beta is None or rho_tilde is None:
                raise ValueError("Heaviside projection filter requires both beta and rho_tilde.")
            dxe = beta * bm.exp(-beta * rho_tilde) + bm.exp(-beta)
            dge[:] = H.matmul(dge * dxe * cell_measure / H.matmul(cell_measure))

        return dge

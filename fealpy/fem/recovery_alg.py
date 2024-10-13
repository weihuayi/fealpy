from typing import Optional
import numpy as np

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from ..functionspace import TensorFunctionSpace

class RecoveryAlg:

    def recovery_estimate(self, uh: TensorLike, method='simple'):
        """
        Calculate the error estimate of gradient recovery for the true solution uh.
        ----------------------------------
        @param uh: The numerical solution uh.
        @param method: The method to compute the recovery estimate. Default is 'simple'.
        @return: Error of the recovery estimate.
        TODO: 向量型的恢复误差估计
        """
        mesh = uh.space.mesh
        rguh = self.grad_recovery(uh, method=method)
        eta = mesh.error(rguh.value, uh.grad_value, power=2, celltype=True) # 计算单元上的恢复型误差
        return eta

    def grad_recovery(self, uh: TensorLike, method='simple'):
        """
        @brief Input a linear finite element function and restore its gradient
        to a piecewise linear finite element space
        ----------------------------------
        @param uh: The numerical solution uh.
        @param method: The method to compute the recovery estimate. Default is 'simple'.
        @return: The recovered gradient of uh.
        """
        space = uh.space
        TD = space.top_dimension()
        GD = space.geo_dimension()
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        bc = bm.array([[1/3]*(TD+1)], dtype=space.ftype)

        guh = uh.grad_value(bc) # (NC, 1, GD)
        
        tensor_space = TensorFunctionSpace(space, shape=(GD, -1))
        rguh = tensor_space.function()
        gval = bm.zeros((gdof, GD), dtype=space.ftype)
        deg = bm.zeros(gdof, dtype=space.ftype)

        if method == 'simple':
            bm.index_add(deg, cell2dof, bm.tensor(1, **bm.context(deg)))
            bm.index_add(gval, cell2dof, guh)
        elif method == 'area_harmonic':
            val = 1.0/space.mesh.entity_measure('cell')
            bm.index_add(deg, cell2dof, val[:, None])
            guh *= val[:, None, None] 
            bm.index_add(gval, cell2dof, guh)
        elif method == 'area':
            val = space.mesh.entity_measure('cell')
            bm.index_add(deg, cell2dof, val[:, None])
            guh *= val[:, None, None] 
            bm.index_add(gval, cell2dof, guh)
        elif method == 'distance':
            ipoints = space.interpolation_points()
            bp = space.mesh.entity_barycenter('cell')
            v = bp[:, None, :] - ipoints[cell2dof, :]
            d = bm.sqrt(bm.sum(v**2, axis=-1))
            guh = bm.einsum('ij...,ij->ij...', guh, d)

            bm.index_add(deg, cell2dof, d)
            bm.index_add(gval, cell2dof, guh)
        elif method == 'distance_harmonic':
            ipoints = space.interpolation_points()
            bp = space.mesh.entity_barycenter('cell')
            v = bp[:, None, :] - ipoints[cell2dof, :]
            d = 1/bm.sqrt(bm.sum(v**2, axis=-1))
            guh = bm.einsum('ij...,ij->ij...', guh, d)

            bm.index_add(deg, cell2dof, d)
            bm.index_add(gval, cell2dof, guh)
        else:
            raise ValueError('Unsupported method: %s' % method)

        gval /= deg[:, None]
        rguh[:] = gval.T.flatten()
        return rguh





import numpy as np
from ..quadrature import FEMeshIntegralAlg

class recovery_alg:
    def __init__(self, space):
        self.space = space
        self.p = space.p
        self.mesh = space.mesh
        self.integralalg = FEMeshIntegralAlg(self.mesh, self.p+3)

    def recovery_estimate(self, uh, method='simple'):
        """
        """
        space = self.space
        rguh = self.grad_recovery(uh, method=method)
        eta = self.integralalg.error(rguh.value, uh.grad_value, power=2, celltype=True) # 计算单元上的恢复型误差
        return eta

    def grad_recovery(self, uh, method='simple'):
        """

        Notes
        -----

        uh 是线性有限元函数，该程序把 uh 的梯度(分片常数）恢复到分片线性连续空间
        中。

        """
        space = self.space
        GD = space.GD
        cell2dof = space.cell_to_dof()
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        p = self.p
        bc = space.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        rguh = space.function(dim=GD)
        if space.doforder == 'sdofs':
            rguh = rguh.T

        if method == 'simple':
            deg = np.bincount(cell2dof.flat, minlength = gdof)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'area':
            measure = self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...', guh, measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'distance':
            ipoints = space.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...', guh, d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'area_harmonic':
            measure = 1/self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...', guh, measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'distance_harmonic':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = 1/np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)
        rguh /= deg.reshape(-1, 1)
        if space.doforder == 'sdofs':
            rguh = rguh.T
        return rguh


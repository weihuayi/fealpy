from fealpy.backend import backend_manager as bm

class GradientReconstruct:
    def __init__(self, mesh):
        self.mesh = mesh

    def AverageGradientreconstruct(self, uh):
        Sf = self.mesh.edge_normal()  
        e2c = self.mesh.edge_to_cell()  
        cell_measure = self.mesh.entity_measure('cell')  
        NC = self.mesh.number_of_cells()
        GD = 2
        uh_i = uh[e2c[:, 0]]  
        uh_j = uh[e2c[:, 1]]  
        uh_f = 0.5 * (uh_i + uh_j) 
        grad_u = bm.zeros((NC, GD)) 
        bm.add.at(grad_u, e2c[:, 0], uh_f[:, None] * Sf)
        mask = e2c[:, 0] != e2c[:, 1]  # 非边界边
        bm.add.at(grad_u, e2c[mask, 1], -uh_f[mask, None] * Sf[mask])
        grad_u /= cell_measure[:, None]  # (NC, 2)
        return grad_u

    def reconstruct(self, uh):
        grad_u = self.AverageGradientreconstruct(uh)
        e2c = self.mesh.edge_to_cell()
        grad_i = grad_u[e2c[:, 0]]  # (NE, 2)
        grad_j = grad_u[e2c[:, 1]]  # (NE, 2)
        grad_f = 0.5 * (grad_i + grad_j)  # (NE, 2)

        return grad_f

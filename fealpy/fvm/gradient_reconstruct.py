from fealpy.backend import backend_manager as bm

class GradientReconstruct():
    def __init__(self,mesh):
        self.mesh = mesh
        
    def gradientreconstruct(self, uh, Sf):
        cell2cell = self.mesh.cell_to_cell()
        S = self.mesh.entity_measure("cell")
        uh_nb = uh[cell2cell]           
        uh_f = 0.5 * (uh[:, None] + uh_nb) 
        grad = bm.sum(uh_f[:, :, None] * Sf, axis=1) / S[:, None]
        grad_nb = grad[cell2cell]  # (NC, 3, 2)
        grad_f = 0.5 * (grad[:, None, :] + grad_nb)  # (NC, 3, 2)
        return grad_f 

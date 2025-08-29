from fealpy.backend import backend_manager as bm

class GradientReconstruct:
    def __init__(self, mesh):
        self.mesh = mesh
        self.Sf = self.mesh.edge_normal()  
        self.e2c = self.mesh.edge_to_cell() 

    def GreenGauss(self, uh):
          
        NC = self.mesh.number_of_cells()
        GD = 2
        uh_i = uh[self.e2c[:, 0]]  
        uh_j = uh[self.e2c[:, 1]]  
        uh_f = 0.5 * (uh_i + uh_j) 
        grad_u = bm.zeros((NC, GD)) 
        bm.add.at(grad_u, self.e2c[:, 0], uh_f[:, None] * self.Sf)
        #mask = e2c[:, 0] != e2c[:, 1]  # 非边界边
        bm.add.at(grad_u, self.e2c[:, 1], -uh_f[:, None] * self.Sf)
        return grad_u

    def AverageGradientreDirichlet(self, uh, gd):
        cell_measure = self.mesh.entity_measure('cell')
        grad_u = self.GreenGauss(uh)
        bdedge = self.mesh.boundary_face_index()
        epoints = self.mesh.entity_barycenter('face')[bdedge, :]
        bdu = gd(epoints)
        bm.add.at(grad_u, self.e2c[bdedge, 0], bdu[:, None] * self.Sf[bdedge, :])
        grad_u /= cell_measure[:, None]  # (NC, 2)
        return grad_u

    def AverageGradientreNeumann(self, uh, gd):
        cell_measure = self.mesh.entity_measure('cell')
        face_measure = self.mesh.entity_measure('face')
        grad_u = self.GreenGauss(uh)
        LNE = self.mesh.number_of_vertices_of_cells()
        bdedge = self.mesh.boundary_face_index()
        # print(bdcell)
        d = 2*cell_measure[self.e2c[bdedge, 0]]/(LNE*face_measure[bdedge])
        bduh = uh[self.e2c[bdedge, 0]]
        gf = gd(self.mesh.entity_barycenter('face')[bdedge, :])
        bdu = bduh + gf * d
        bm.add.at(grad_u, self.e2c[bdedge, 0], bdu[:, None] * self.Sf[bdedge, :])
        grad_u /= cell_measure[:, None]  # (NC, 2)
        return grad_u

    def reconstruct(self, grad_u):
        e2c = self.mesh.edge_to_cell()
        grad_i = grad_u[e2c[:, 0]]  # (NE, 2)
        grad_j = grad_u[e2c[:, 1]]  # (NE, 2)
        grad_f = 0.5 * (grad_i + grad_j)  # (NE, 2)

        return grad_f
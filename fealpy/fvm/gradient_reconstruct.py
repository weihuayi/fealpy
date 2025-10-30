from fealpy.backend import backend_manager as bm

class GradientReconstruct:
    def __init__(self, mesh):
        self.mesh = mesh
        self.Sf = self.mesh.edge_normal()  
        self.e2c = self.mesh.edge_to_cell() 

    def GreenGauss(self, U):
        GD = U[..., None].shape[1]
        NC = self.mesh.number_of_cells()
        if GD == 1:
            grad_U = bm.zeros((NC, 2))
            uh_i = U[self.e2c[:, 0]]  
            uh_j = U[self.e2c[:, 1]]  
            uh_f = 0.5 * (uh_i + uh_j) 
            bm.add.at(grad_U, self.e2c[:, 0], uh_f[:, None] * self.Sf)
            bm.add.at(grad_U, self.e2c[:, 1], -uh_f[:, None] * self.Sf)
        elif GD == 2:
            grad_u = bm.zeros((NC, 2))
            uh_i = U[self.e2c[:, 0],0]  
            uh_j = U[self.e2c[:, 1],0]  
            uh_f = 0.5 * (uh_i + uh_j) 
            bm.add.at(grad_u, self.e2c[:, 0], uh_f[:, None] * self.Sf)
            bm.add.at(grad_u, self.e2c[:, 1], -uh_f[:, None] * self.Sf)
            grad_v = bm.zeros((NC, 2))
            vh_i = U[self.e2c[:, 0],1]  
            vh_j = U[self.e2c[:, 1],1]  
            vh_f = 0.5 * (vh_i + vh_j) 
            bm.add.at(grad_v, self.e2c[:, 0], vh_f[:, None] * self.Sf)
            bm.add.at(grad_v, self.e2c[:, 1], -vh_f[:, None] * self.Sf)
            grad_U = bm.stack([grad_u,grad_v], axis=1)
        return grad_U

    def test(self, U):
        cell_measure = self.mesh.entity_measure('cell')
        grad_U = self.GreenGauss(U)
        grad_U /= cell_measure[:, None]  # (NC, 2)
        return grad_U

    def AverageGradientreDirichlet(self, U, gd):
        cell_measure = self.mesh.entity_measure('cell')
        grad_U = self.GreenGauss(U)
        bdedge = self.mesh.boundary_face_index()
        epoints = self.mesh.entity_barycenter('face')[bdedge, :]
        bdu = gd(epoints)
        GD = U[..., None].shape[1]
        if GD == 1:
            bm.add.at(grad_U, self.e2c[bdedge, 0], bdu[:, None] * self.Sf[bdedge, :])
            grad_U /= cell_measure[:, None]  # (NC, 2)
        elif GD == 2:
            bm.add.at(grad_U[:,0,:], self.e2c[bdedge, 0], bdu[:, 0, None] * self.Sf[bdedge, :])
            bm.add.at(grad_U[:,1,:], self.e2c[bdedge, 0], bdu[:, 1, None] * self.Sf[bdedge, :])
            grad_U /= cell_measure[:, None, None]
        return grad_U

    def AverageGradientreNeumann(self, uh, gd):
        cell_measure = self.mesh.entity_measure('cell')
        face_measure = self.mesh.entity_measure('face')
        grad_u = self.GreenGauss(uh)
        LNE = self.mesh.number_of_vertices_of_cells()
        bdedge = self.mesh.boundary_face_index()
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
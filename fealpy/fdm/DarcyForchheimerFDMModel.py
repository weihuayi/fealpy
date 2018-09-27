import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack


class DarcyForchheimerFDMModel():
    def __init__(self):
    	self.pde = pde
        self.mesh = mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        self.uh = np.zeros(NE, dtype=mesh.ftype)
        self.ph = np.zeros(NC, dtype=mesh.ftype)
        self.uI = np.zeros(NE, dtype=mesh.ftype) 
        
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        bc = mesh.entity_barycenter('edge')

        self.uI[isYDEdge] = pde.velocity_x(bc[isYDEdge])
        self.uI[isXDEdge] = pde.velocity_y(bc[isXDEdge]) 
        pc = mesh.entity_barycenter('cell')
        self.pI = pde.pressure(pc)

        self.ph[0] = self.pI[0]
        pass

    def Qu(self):
        pass
    def Qv(self):
        pass

    def get_left_matrix(self,nx,ny,NC,NE):
   


        return A

    def get_right_vector(self):
        pass

    def solve(self):
        pass
 
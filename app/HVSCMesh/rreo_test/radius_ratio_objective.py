from scipy.sparse import csr_matrix

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesh.opt import SumObjective

class RadiusRatioSumObjective(SumObjective):
    def __init__(self, mesh_quality):
        super().__init__(mesh_quality)

    def hess(self,x:TensorLike,funtype=0):
        cell = self.mesh_quality.mesh.entity('cell')
        NN = self.mesh_quality.mesh.number_of_nodes()
        NC = self.mesh_quality.mesh.number_of_cells()
        TD = self.mesh_quality.mesh.TD

        if self.mesh_quality.mesh.TD == 2:
            A,B = self.mesh_quality.hess(x,funtype)
            I = bm.broadcast_to(cell[:,:,None],(NC,3,3))
            J = bm.broadcast_to(cell[:,None,:],(NC,3,3))
            A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))
            B = csr_matrix((B.flat,(I.flat,J.flat)),shape=(NN,NN))
            return (A,B)

        elif self.mesh_quality.mesh.TD == 3:
            A,B0,B1,B2 = self.mesh_quality.hess(x,funtype)
            I = bm.broadcast_to(cell[:,:,None],(NC,4,4))
            J = bm.broadcast_to(cell[:,None,:],(NC,4,4))
            A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))
            B0 = csr_matrix((B0.flat, (I.flat, J.flat)), shape=(NN, NN))
            B1 = csr_matrix((B1.flat, (I.flat, J.flat)), shape=(NN, NN))
            B2 = csr_matrix((B2.flat, (I.flat, J.flat)), shape=(NN, NN))
            return (A,B0,B1,B2)



































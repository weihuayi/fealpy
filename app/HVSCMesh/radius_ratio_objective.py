from scipy.sparse import csr_matrix

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesh.opt import SumObjective

class RadiusRatioSumObjective(SumObjective):
    def __init__(self, mesh_quality):
        super().__init__(mesh_quality)

    def fun_with_grad(self,x:TensorLike):
        if len(x.shape) == 1:
            TD = self.mesh_quality.mesh.TD
            NN = self.mesh_quality.mesh.number_of_nodes()
            x = x.reshape(TD,-1).T
            if x.shape[0]!=NN:
                isBdNode = self.mesh_quality.mesh.boundary_node_flag()
                node0 = self.mesh_quality.mesh.entity('node')
                x0 = bm.full_like(node0,0.0)
                x0[~isBdNode,:] = x
                x0[isBdNode,:] = node0[isBdNode,:]
                x = x0
                return self.fun(x),self.jac(x,return_free=True).T.flatten()
            return self.fun(x), self.jac(x).T.flatten()
        return self.fun(x), self.jac(x).T.flatten()
    
    def jac(self,x:TensorLike,return_free = False):
        if return_free is False:
            TD = self.mesh_quality.mesh.TD
            NN = self.mesh_quality.mesh.number_of_nodes()
            cell = self.mesh_quality.mesh.entity('cell')
            grad = self.mesh_quality.jac(x)
            jacobi = bm.zeros((NN,TD))
            for i in range(TD):
                bm.index_add(jacobi[:,i],cell.flatten(),grad[:,:,i].flatten())
            return jacobi
        else:
            isFreeNode = ~self.mesh_quality.mesh.boundary_node_flag()
            TD = self.mesh_quality.mesh.TD
            NN = self.mesh_quality.mesh.number_of_nodes()
            cell = self.mesh_quality.mesh.entity('cell')
            grad = self.mesh_quality.jac(x)
            jacobi = bm.zeros((NN,TD))
            for i in range(TD):
                bm.index_add(jacobi[:,i],cell.flatten(),grad[:,:,i].flatten())
            jacobi = jacobi[isFreeNode,:]
            return jacobi

    
    def hess(self,x:TensorLike):
        cell = self.mesh_quality.mesh.entity('cell')
        NN = self.mesh_quality.mesh.number_of_nodes()
        NC = self.mesh_quality.mesh.number_of_cells()
        TD = self.mesh_quality.mesh.TD

        if self.mesh_quality.mesh.TD == 2:
            A,B = self.mesh_quality.hess(x)
            I = bm.broadcast_to(cell[:,:,None],(NC,3,3))
            J = bm.broadcast_to(cell[:,None,:],(NC,3,3))
            A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))
            B = csr_matrix((B.flat,(I.flat,J.flat)),shape=(NN,NN))
            return (A,B)

        elif self.mesh_quality.mesh.TD == 3:
            A,B0,B1,B2 = self.mesh_quality.hess(x)
            I = bm.broadcast_to(cell[:,:,None],(NC,4,4))
            J = bm.broadcast_to(cell[:,None,:],(NC,4,4))
            A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))
            B0 = csr_matrix((B0.flat, (I.flat, J.flat)), shape=(NN, NN))
            B1 = csr_matrix((B1.flat, (I.flat, J.flat)), shape=(NN, NN))
            B2 = csr_matrix((B2.flat, (I.flat, J.flat)), shape=(NN, NN))
            return (A,B0,B1,B2)































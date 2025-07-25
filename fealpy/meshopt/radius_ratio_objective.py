from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import CSRTensor,COOTensor
from .opt import SumObjective


class RadiusRatioSumObjective(SumObjective):
    '''
    Objective function for optimizing the sum of radius ratio quality metrics 
    over a mesh.Handles both 2D (triangular) and 3D (tetrahedral) meshes, 
    providing function evaluation, gradients, Hessians, and preconditioners 
    for optimization algorithms.
    Parameters:
        mesh_quality(RadiusRatioQuality): Pre-initialized quality metric 
        calculator for the target mesh
    '''
    def __init__(self, mesh_quality):
        '''
        Initializes the optimization objective function and precomputes 
        preconditioner components.
        Parameters:
            mesh_quality(RadiusRatioQuality): Pre-initialized mesh quality 
            metric calculator
        '''
        super().__init__(mesh_quality)
        isFreeNode = ~mesh_quality.mesh.boundary_node_flag()
        node = mesh_quality.mesh.entity('node')
        self.x0 = node[isFreeNode,:].T.flatten()
        A = self.hess(mesh_quality.mesh.node)
        row = A.row
        col = A.col
        data = A.data
        rowtag = isFreeNode[row]
        coltag = isFreeNode[col]
        datatag = rowtag & coltag
        self.datatag = datatag
        data = data[datatag]
        row = row[datatag]
        col = col[datatag]
        freetag = bm.cumsum(isFreeNode,axis=0,dtype=bm.int64) - bm.ones(len(isFreeNode),
                                                                        dtype=bm.int64)
        row = freetag[row]
        col = freetag[col]
        NI = bm.sum(isFreeNode)
        self.NI = NI
        ND = len(data)
        self.ND = ND
        newdata = bm.zeros(3*ND,dtype=data.dtype)
        newdata = bm.set_at(newdata,(slice(0,ND)),data) 
        newdata = bm.set_at(newdata,(slice(ND,2*ND)),data)
        newdata = bm.set_at(newdata,(slice(2*ND,None)),data)
        '''
        newdata[:ND] = data
        newdata[ND:2*ND] = data
        newdata[2*ND:] = data
        '''
        newrow = bm.zeros(3*ND,dtype=row.dtype)
        newrow = bm.set_at(newrow, (slice(0,ND)),row)
        newrow = bm.set_at(newrow, (slice(ND,2*ND)),row + NI)
        newrow = bm.set_at(newrow, (slice(2*ND,None)),row + 2*NI)
        '''
        newrow[:ND] = row
        newrow[ND:2*ND] = row + NI
        newrow[2*ND:] = row + 2*NI
        '''
        newcol = bm.zeros(3*ND,dtype=col.dtype)
        newcol = bm.set_at(newcol, (slice(0,ND)),col) 
        newcol = bm.set_at(newcol, (slice(ND,2*ND)),col + NI)
        newcol = bm.set_at(newcol, (slice(2*ND,None)),col + 2*NI)
        '''
        newcol[:ND] = col
        newcol[ND:2*ND] = col + NI
        newcol[2*ND:] = col + 2*NI
        '''
        self.indice = bm.stack([newrow,newcol],axis=0)
        self.P = COOTensor(self.indice,newdata,spshape=(3*NI,3*NI))

    def fun_with_grad(self,x: TensorLike):
        '''
        Computes both the objective function value and its gradient simultaneously.
        Parameters:
            x(TensorLike): Node coordinates either as:
                - Flattened 1D tensor of free node positions
                - Full 2D tensor of shape (NN, GD)
        Returns:
            Tuple[TensorLike, TensorLike]
                Tuple containing:
                1. Function value (scalar)
                2. Gradient vector (flattened 1D tensor of free node gradients)
        '''
        if len(x.shape) == 1:
            TD = self.mesh_quality.mesh.TD
            NN = self.mesh_quality.mesh.number_of_nodes()
            x = x.reshape(TD,-1).T
            if x.shape[0]!=NN:
                isBdNode = self.mesh_quality.mesh.boundary_node_flag()
                node0 = self.mesh_quality.mesh.entity('node')
                x0 = bm.full_like(node0,0.0)
                #x0[~isBdNode,:] = x
                x0 = bm.set_at(x0, (~isBdNode), x)
                #x0[isBdNode,:] = node0[isBdNode,:]
                x0 = bm.set_at(x0, (isBdNode), node0[isBdNode,:])
                x = x0
                return self.fun(x),self.jac(x,return_free=True).T.flatten()
            return self.fun(x), self.jac(x).T.flatten()
        return self.fun(x), self.jac(x).T.flatten()
    
    def jac(self,x:TensorLike,return_free = False):
        '''
        Computes the Jacobian of the objective function.
        Parameters:
            x(TensorLike): Node coordinates tensor of shape (NN, GD)
            return_free(bool, optional): Whether to return only gradients for 
            free (non-boundary) nodes
        Returns:
            TensorLike
                - If return_free=False: Full gradient matrix of shape (NN, GD)
                - If return_free=True: Gradient matrix for free nodes only
        '''
        if return_free is False:
            TD = self.mesh_quality.mesh.TD
            NN = self.mesh_quality.mesh.number_of_nodes()
            cell = self.mesh_quality.mesh.entity('cell')
            grad = self.mesh_quality.jac(x)
            jacobi = bm.zeros((NN,TD))
            for i in range(TD):
                tem_jacobi = bm.zeros(NN,dtype=grad.dtype)
                tem_jacobi = bm.index_add(tem_jacobi,cell.flatten(),grad[:,:,i].flatten())
                jacobi = bm.set_at(jacobi, (slice(None),i), tem_jacobi)
                #bm.index_add(jacobi[:,i],cell.flatten(),grad[:,:,i].flatten())
            return jacobi
        else:
            isFreeNode = ~self.mesh_quality.mesh.boundary_node_flag()
            TD = self.mesh_quality.mesh.TD
            NN = self.mesh_quality.mesh.number_of_nodes()
            cell = self.mesh_quality.mesh.entity('cell')
            grad = self.mesh_quality.jac(x)
            jacobi = bm.zeros((NN,TD),dtype=grad.dtype)
            for i in range(TD):
                tem_jacobi = bm.zeros(NN,dtype=grad.dtype)
                #bm.set_at(jacobi, (slice(None,i)),bm.index_add(jacobi[:,i],cell.flatten(),grad[:,:,i].flatten()))
                #bm.index_add(jacobi[:,i],cell.flatten(),grad[:,:,i].flatten())
                tem_jacobi = bm.index_add(tem_jacobi,cell.flatten(),grad[:,:,i].flatten())
                jacobi = bm.set_at(jacobi, (slice(None),i), tem_jacobi)
                #jacobi = bm.set_at(jacobi,(cell.flatten(),i),grad[:,:,i].flatten())
            jacobi = jacobi[isFreeNode,:]
            return jacobi
    
    def hess(self,x:TensorLike):
        '''
        Computes the Hessian approximation of the objective function.
        Parameters:
            x(TensorLike): Node coordinates tensor of shape (NN, GD)
        Returns:
            Union[Tuple[COOTensor, COOTensor], COOTensor]
                Hessian representation:
                - For 2D: Tuple of two COOTensors (A, B) for second derivatives
                - For 3D: Single COOTensor containing Hessian approximation
        '''
        cell = self.mesh_quality.mesh.entity('cell')
        NN = self.mesh_quality.mesh.number_of_nodes()
        NC = self.mesh_quality.mesh.number_of_cells()
        TD = self.mesh_quality.mesh.TD

        if self.mesh_quality.mesh.TD == 2:
            A,B = self.mesh_quality.hess(x)
            I = bm.broadcast_to(cell[:,:,None],(NC,3,3))
            J = bm.broadcast_to(cell[:,None,:],(NC,3,3))
            indice = bm.stack([I.flatten(),J.flatten()],axis=0)
            data = A.flatten()
            A = COOTensor(I.flat,J.flat,A.flat,spshape=(NN,NN))
            B = COOTensor(I.flat,J.flat,B.flat,spshape=(NN,NN))
            return (A,B)

        elif self.mesh_quality.mesh.TD == 3:
            A = self.mesh_quality.hess(x)
            I = bm.broadcast_to(cell[:,:,None],(NC,4,4))
            J = bm.broadcast_to(cell[:,None,:],(NC,4,4))
            indice = bm.stack([I.flatten(),J.flatten()],axis=0)
            data = A.flatten()
            A = COOTensor(indice,data,spshape=(NN,NN))
            return A

    def preconditioner(self,x:TensorLike):
        """
        Applies preconditioning using the Conjugate Gradient method.
        Parameters:
            x(TensorLike): Input vector to precondition
        Returns:
            TensorLike
                Preconditioned vector
        """
        from ..solver import cg

        r = cg(self.P,x)
        return r

    def update_preconditioner(self,x:TensorLike):
        '''
        Updates the preconditioner matrix based on current node positions.
        Parameters:
            x(TensorLike): Current free node positions as a flattened 1D tensor
        Returns:
            COOTensor
                Updated preconditioner matrix in COO sparse format
        '''
        isFreeNode = ~self.mesh_quality.mesh.boundary_node_flag()
        node0 = bm.copy(self.mesh_quality.mesh.entity('node'))
        NI = self.NI
        node0 = bm.set_at(node0, (isFreeNode, 0), x[:NI])
        node0 = bm.set_at(node0, (isFreeNode, 1), x[NI:2*NI])
        node0 = bm.set_at(node0, (isFreeNode, 2), x[2*NI:])
        '''
        node0[isFreeNode,0] = x[:NI]
        node0[isFreeNode,1] = x[NI:2*NI]
        node0[isFreeNode,2] = x[2*NI:]
        '''
        A = self.hess(node0)
        data = A.data[self.datatag]
        ND = self.ND
        newdata = bm.zeros(3*ND,dtype=data.dtype)
        newdata = bm.set_at(newdata, (slice(0,ND)), data)
        newdata = bm.set_at(newdata, (slice(ND,2*ND)), data)
        newdata = bm.set_at(newdata, (slice(2*ND,None)), data)
        '''
        newdata[:ND] = data 
        newdata[ND:2*ND] = data
        newdata[2*ND:] = data
        '''
        self.P = COOTensor(self.indice,newdata,spshape=(3*NI,3*NI))
        return self.P






























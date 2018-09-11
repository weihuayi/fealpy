import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat
from fealpy.fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
class DarcyFDMModel():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

    def get_left_matrix(self):

        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        itype = mesh.itype
        ftype = mesh.ftype

        idx = np.arange(NE)
        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        mu = self.pde.mu
        k = self.pde.k
        A11 = mu/k*eye(NE, NE, dtype=ftype)
        print('A11',A11)


        edge2cell = mesh.ds.edge_to_cell()
        I, = np.nonzero(~isBDEdge & isYDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/mesh.hx

        A12 = coo_matrix((data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((-data, (I, L)), shape=(NE, NC))

        I, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/mesh.hy
        A12 += coo_matrix((-data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((data, (I, L)), shape=(NE, NC))
        A12 = A12.tocsr()
        print("A12",A12)

        
        cell2edge = mesh.ds.cell_to_cedge()
        I = np.arange(NC, dtype=itype)
        data = np.ones(NC, dtype=ftype)
        A21 = coo_matrix((data/mesh.hx, (I, cell2edge[:, 1])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-data/mesh.hx, (I, cell2edge[:, 3])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((data/mesh.hy, (I, cell2edge[:, 2])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-data/mesh.hy, (I, cell2edge[:, 0])), shape=(NC, NE), dtype=ftype)
        A21 = A21.tocsr()

        A = bmat([(A11, A12), (A21, None)], format='csr', dtype=ftype)

        return A

    def get_right_vector(self,nx,ny):
        mesh = self.mesh
        node = mesh.node
        itype = mesh.itype
        ftype = mesh.ftype

        NN = mesh.number_of_nodes()  
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        dn = mesh.ds.edge_to_node()
        

        b = np.zeros(NE+NC, dtype=ftype)
        
        
        dn1 = dn[:int(NE/2),:]
        nodex = (node[dn1[:,1],:] + node[dn1[:,0],:])/2

        dn2 = dn[int(NE/2):,:]
        nodey = (node[dn2[:,1],:] + node[dn2[:,0],:])/2
        
       
        nodep = np.zeros((NC,2), dtype=ftype)
        d = int(NE/2/(ny+1))
        for i in range(d):
            nodep[i*ny:(i+1)*ny,0] = nodey[i*(ny+1)+1:(i+1)*(ny+1)]
        nodep[:,1] = nodex[ny:,1]
        b[NE:] = self.pde.source(nodep)
        return b


    def solve(self,A,b,nx,ny):
        mesh = self.mesh
        x = np.zeros((NE+NC,), dtype=np.ftype)


        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag() 
        

        # Modify the right hand
        dn1 = dn[:int(NE/2),:]
        nodex = (node[dn1[:,1],:] + node[dn1[:,0],:])/2
        idx1, = np.nonzero(isBDEdge & isYDEdge)
        x[idx1] = self.pde.velocity(nodex[idx1,:])[:,0]

        dn2 = dn[int(NE/2):,:]
        nodey = (node[dn2[:,1],:] + node[dn2[:,0],:])/2
        idx2, = np.nonzero(isBDEdge & isXDEdge)
        x[idx2] = self.pde.velocity(nodey[idx2,:])[:,1]


        nodep = np.zeros((NC,2), dtype=ftype)
        d = int(NE/2/(ny+1))
        for i in range(d):
            nodep[i*ny:(i+1)*ny,0] = nodey[i*(ny+1):(i+1)*(ny+1)]
        nodep[:,1] = nodex[ny:,1]

        x[NE] = self.pde.pressure(nodeyy[0,:])
        b = b - A@x

        # Modify matrix
        bdIdx = np.zeros((A.shape[0],),dtype = itype)
        bdIdx[~isBDEdge] = 1
        bdIdx[NE] = 0

        Tbd = spdiags(bdIdx, 0, A.shape[0],A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0],A.shape[0])
        AD = T@A@T + Tbd

        # solve
        isBDNode1 = np.vstack((idx1,idx2)).flatten()
        isBDNode = np.hstack((isBDNode1,NE))
        freeNode = [i for i in range(NE+NC) if i not in isBDNode]
        x[freeNode] = spsolve(AD[isBDNode],b[isBDNode])
        

        return x

    def get_L2_error(self, x, nx, ny):
        mesh = self.mesh

        NE = mesh.number_of_edges() 
        NC = mesh.number_of_cells()


        U = self.pde.velocity
        P = self.pde.pressure

        u = x[:NE]
        p = x[NE:]
        erruL2 = np.sqrt(sum(hx*hy*(U[:] - u)**2))
        errpL2 = np.sqrt(sum(hx*hy*(P - p)**2))

        return erruL2, errpL2

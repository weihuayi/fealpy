import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack


class DarcyFDMModel():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

    def get_left_matrix(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()

        idx = np.arange(NE)
        edge2cell = mesh.ds.edge_to_cell()
        isIntEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        idx = idx[isIntEdge]


        A11 = csr_matrix((data1, (idx, idx)),shape = (NE, NE))



        N3 = np.arange((NE/2),dtype = np.int).reshape(4,5)[:,:-1].flatten()
        N4 = np.arange((NE/2+1),dtype = np.int)[1:].reshape(4,5)[:,:-1].flatten()
        col2 = [rv for r in zip(N3,N4) for rv in r]
        A12 = csr_matrix((data1, (row1, col2)),shape = (NC, NE/2))
        A13 = csr_matrix((NC, NC),dtype = np.int)
        A1 = hstack((A11, A12, A13))

        row2 = np.arange(4,NE/2)
        data2 = mu*nx/k*np.array([1]*(int(NE/2-ny)))
        A21 = csr_matrix((data2,(row2,row2)),shape = (NE/2, NE/2))
        A22 = csr_matrix((NE/2, NE/2),dtype = np.int)
        row3 = [val for val in np.arange(ny,NE/2-ny) for i in range(2)]
        N5 = np.arange(NE/2-2*ny)
        N6 = np.arange(ny,NE/2-ny)
        col3 = [rv for r in zip(N5,N6) for rv in r]
        data3 = np.array([-1,1 ]*(int(NE/2-2*ny)))
        A23 = csr_matrix((data3,(row3,col3)),shape = (NE/2, NC))
        A2 = hstack((A21, A22, A23))
        A31 = csr_matrix((NE/2, NE/2),dtype = np.int)
        row4 = np.arange((NE/2),dtype = np.int).reshape(4,5)[:,1:].flatten()
        A32 = csr_matrix((data2,(row4,row4)),shape = (NE/2, NE/2))
        N7 = np.arange((NE/2),dtype = np.int).reshape(4,5)[:,1:-1].flatten()
        row5 = [val for val in N7 for i in range(2)]
        N8 = np.arange((NC),dtype = np.int).reshape(4,4)[:,:-1].flatten()
        N9 = np.arange((NC+1),dtype = np.int)[1:].reshape(4,4)[:,:-1].flatten()
        col4 = [rv for r in zip(N8,N9) for rv in r]
        data4 = np.array([-1,1 ]*int(NC-nx))
        A33 = csr_matrix((data4,(row5,col4)),shape = (NE/2, NC))
        A3 = hstack((A31, A32, A33))
        A = vstack((A1,A2,A3)).toarray()

        return A

    def get_right_vector(self):
        pass

    def solve(self):
        pass

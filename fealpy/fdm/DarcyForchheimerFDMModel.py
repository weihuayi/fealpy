import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack


class DarcyForchheimerFDMModel():
    def __init__(self):
    	self.fdmspace = 
        pass

    def get_left_matrix(self,nx,ny,NC,NE):
    	row1 = [val for val in np.arange(NC) for i in range(2)]
    	N1 = np.arange(NE/2-ny)
    	N2 = np.arange(ny,NE/2)
    	col1 = [rv for r in zip(N1,N2) for rv in r]
    	data1 = np.array([-1,1 ]*NC)
    	A11 = csr_matrix((data1, (row1, col1)),shape = (NC, NE/2)).toarray()
    	N3 = np.arange((NE/2),dtype = np.int).reshape(4,5)[:,:-1].flatten()
    	N4 = np.arange((NE/2+1),dtype = np.int)[1:].reshape(4,5)[:,:-1].flatten()
    	col2 = [rv for r in zip(N3,N4) for rv in r]
    	A12 = csr_matrix((data1, (row1, col2)),shape = (NC, NE/2)).toarray()
    	A13 = csr_matrix(NC, NC).toarray()
    	A1 = hstack((A11, A12, A13)).toarray()

    	row2 = np.arange(4,NE/2)
    	data2 = mu*nx/k*np.array([1]*(int(NE/2-ny)))
    	A21 = csr_matrix((data2,(row2,row2)),shape = (NE/2, NE/2)).toarray()
    	A22 = csr_matrix(NE/2, NE/2).toarray()
    	row3 = [val for val in np.arange(4,NE/2-ny) for i in range(2)]
    	N5 = np.arange(NE/2-2*ny)
    	N6 = np.arange(ny,NE/2-ny)
    	col3 = [rv for r in zip(N5,N6) for rv in r]
    	data3 = np.array([-1,1 ]*(int(NE/2-2*ny)))
    	A23 = csr_matrix((data2,(row3,col3)),shape = (NE/2, NC)).toarray()
    	A2 = hstack((A21, A22, A23)).toarray()

    	A31 = csr_matrix(NE/2, NE/2).toarray()
    	row4 = np.arange((NE/2),dtype = np.int).reshape(4,5)[:,1:].flatten()
    	A32 = csr_matrix((data2,(row4,row4)),shape = (NE/2, NE/2)).toarray()
    	row5 = np.arange((NE/2),dtype = np.int).reshape(4,5)[:,1:-1].flatten()
    	N7 = np.arange((NC),dtype = np.int).reshape(4,4)[:,:-1].flatten()
    	N8 = np.arange((NC+1),dtype = np.int)[1:].reshape(4,4)[:,:-1].flatten()
    	col4 = [rv for r in zip(N7,N8) for rv in r]
    	data4 = np.array([-1,1 ]*int(NC-nx))
    	A33 = csr_matrix((data4,(row5,col4)),shape = (NE/2, NC)).toarray()


        return A

    def get_right_vector(self):
        pass

    def solve(self):
        pass
 
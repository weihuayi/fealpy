import numpy as np
from scipy.sparse import csr_matrix

def scaleCoor(realp):
    center = np.mean(realp,axis=1)

    pn = realp.shape[0]

    diff = realp - center*np.ones((pn,2))

    h = 0.1*np.max(np.sqrt(np.sum(diff**2,axis=1)))

    refp = diff/h
    return refp, center, h


class FEMFunctionRecoveryAlg():
    def __init__(self):
        pass

    def simple_average(self, uh):
        V = uh.V
        mesh = V.mesh
        GD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        valence = node2cell.sum(axis=1)

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)
        rguh = V.function(dim=GD)
        rguh[:] = np.asarray(node2cell@guh)/valence.reshape(-1, 1)
        return rguh

    def area_average(self, uh):
        V = uh.V
        mesh = V.mesh
        GD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        area = mesh.area()
        asum = node2cell@area

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        rguh = V.function(dim=GD)
        rguh[:] = np.asarray(node2cell@(guh*area.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh

    def harmonic_average(self, uh):
        V = uh.V
        mesh = V.mesh
        GD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        inva = 1/mesh.area()
        asum = node2cell@inva

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        rguh = V.function(dim=GD)
        rguh[:] = np.asarray(node2cell@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh
    


    def SCR(self,uh):
        V = uh.V 
        mesh = V.mesh
        GD = mesh.geo_dimension()
        rguh = V.function(dim=GD)
          
        cell = mesh.ds.cell
        node = mesh.node
        print(node)
        
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        
        row = np.arange(NC).repeat(3)
        col = cell.flatten()
        data = np.ones(NC*3)
        t2p = csr_matrix((data, (row, col)), shape=(NC,NN)).toarray()
        p2t = t2p.T
        p2p = p2t@t2p
        
        for i in range(N):
            np = [j for (j,val) in enumerate(p2p[:,i]) if val>0]
            temp0 = node[np,:]
                 
            #调用scaleCoor()
            tempx,_,h = scaleCoor(temp0)

            tempp = uh[np,:]
            tempn = tempx.shape[0]
            X = np.ones((tempn,3))
            X[:,1:3] = tempx
            
            coefficient = np.linalg.solve(X.T@X,X.T@tempp[:,0])        
                    
            rguh[i,0] = coefficient[1]/h
            rguh[i,1] = coefficient[2]/h
        return rguh


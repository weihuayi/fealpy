import numpy as np
from scipy.sparse import csr_matrix
from fealpy.quadrature import TriangleQuadrature

def scaleCoor(realp):
    center = np.mean(realp,axis=0)

    pn = realp.shape[0]

    diff = realp - center*np.ones((pn,2))

    h = np.max(np.sqrt(np.sum(diff**2,axis=1)))
    refp = diff/h
    return refp, center, h


class FEMFunctionRecoveryAlg():
    def __init__(self):
        pass

    def integrator(self, k):
        return TriangleQuadrature()

    def simple_average(self, uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()

        node2cell = mesh.ds.node_to_cell()
        valence = node2cell.sum(axis=1)

        if TD == 2:
            bc = np.array([1/3]*3, dtype=np.float)
        elif TD == 3:
            bc = np.array([1/4]*4, dtype=np.float)

        guh = uh.grad_value(bc)
        rguh = space.function(dim=GD)
        rguh[:] = np.asarray(node2cell@guh)/valence.reshape(-1, 1)
        return rguh

    def area_average(self, uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        TD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        measure = mesh.entity_measure('cell')
        asum = node2cell@measure

        if TD == 2:
            bc = np.array([1/3]*3, dtype=np.float)
        elif TD == 3:
            bc = np.array([1/4]*4, dtype=np.float)

        guh = uh.grad_value(bc)

        rguh = space.function(dim=GD)
        rguh[:] = np.asarray(node2cell@(guh*area.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh

    def harmonic_average(self, uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()

        node2cell = mesh.ds.node_to_cell()
        inva = 1/mesh.entity_measure('cell')
        asum = node2cell@inva

        if TD == 2:
            bc = np.array([1/3]*3, dtype=np.float)
        elif TD == 3:
            bc = np.array([1/4]*4, dtype=np.float)

        guh = uh.grad_value(bc)
        rguh = space.function(dim=GD)
        rguh[:] = np.asarray(node2cell@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh

    def distance_harmonic_average(self, uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        NN = mesh.number_of_nodes()

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        bp = mesh.entity_barycenter('cell')
        v = bp[:, np.newaxis, :] - node[cell, :]
        d = 1/np.sqrt(np.sum(v**2, axis=-1))
        dsum = np.bincount(cell.flat, weights=d.flat, minlength=NN)
        rguh = space.function(dim=GD)
        for i in range(GD):
            val = guh[:, [i]]*d
            rguh[:, i] = np.bincount(cell.flat, weights=val.flat, minlength=NN)/dsum
        return rguh

    def angle_average(self, uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        NN = mesh.number_of_nodes()

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)


        cell = mesh.entity('cell')
        angle = mesh.angle()
        asum = np.bincount(cell.flat, weights=angle.flat, minlength=NN)
        rguh = space.function(dim=GD)
        for i in range(GD):
            val = guh[:, [i]]*angle
            rguh[:, i] = np.bincount(cell.flat, weights=val.flat, minlength=NN)/asum
        return rguh


    def SCR(self,uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        rguh = space.function(dim=GD)
          
        cell = mesh.ds.cell
        node = mesh.node
        
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        
        row = np.arange(NC).repeat(3)
        col = cell.flatten()
        data = np.ones(NC*3)
        t2p = csr_matrix((data, (row, col)), shape=(NC,NN)).toarray()
        p2t = t2p.T
        p2p = p2t@t2p
        
        for i in range(NN):
            np1, =np.nonzero(p2p[:,i]) 
            temp0 = node[np1,:]
            #调用scaleCoor()
            tempx,_,h = scaleCoor(temp0)


            tempp = uh[np1]
            tempn = tempx.shape[0]
            X = np.ones((tempn,3))
            X[:,1:3] = tempx
            
            coefficient = np.linalg.solve(X.T@X,X.T@tempp)        
                    
            rguh[i,0] = coefficient[1]/h
            rguh[i,1] = coefficient[2]/h
        return rguh

    def ZZ(self, uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        rguh = space.function(dim=GD)
       
          

        cell = mesh.ds.cell
        node = mesh.node

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        isBdNodes = mesh.ds.boundary_node_flag()
        xnode = mesh.barycenter()

        row = np.arange(NC).repeat(3)
        col = cell.flatten()
        data = np.ones(NC*3)
        t2p = csr_matrix((data, (row, col)), shape=(NC, NN)).toarray()
        p2t = t2p.T
        p2p = p2t@t2p
        
        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc) 
        
        for i in range(NN):
            if isBdNodes[i]:
                np1, = np.nonzero(p2p[:, i])
                ip = np1[~isBdNodes[np1]]
                ipn = ip.shape[0]
                if ipn == 0:
                    ne, = np.nonzero(t2p[:, i])
                    tempp = guh[ne, :]
                    rguh[i, :] = np.mean(tempp, axis=0)
                    #rguh[i, :] = rguh[i, :]
                else:
                    for k in range(ipn):
                        np2, = np.nonzero(t2p[:, ip[k]])
                        temp0 = xnode[np2,:]
                        tempx, center, h = scaleCoor(temp0)
                        tempp = guh[np2, :]
                        tempn = tempx.shape[0]
                        X = np.ones((tempn, 3))
                        X[:, 1:3] = tempx
                        coefficient1 = np.linalg.solve(X.T@X,X.T@tempp[:,0])
                        rguh[i,0] = rguh[i,0] + node[i,:]@coefficient1[1:3]/h + coefficient1[0] - center@coefficient1[1:3]/h
                        coefficient2 = np.linalg.solve(X.T@X,X.T@tempp[:,1])
                        rguh[i,1] = rguh[i,1] + node[i,:]@coefficient2[1:3]/h + coefficient2[0] - center@coefficient2[1:3]/h
                    rguh[i, :] = rguh[i, :]/ipn
            else:
                ne, = np.nonzero(t2p[:, i])
                temp0 =xnode[ne, :]
                tempx,center,h = scaleCoor(temp0)
                tempp = guh[ne, :]
                tempn = tempx.shape[0]
                X = np.ones((tempn, 3))
                X[:, 1:3] = tempx
                coefficient3 = np.linalg.solve(X.T@X,X.T@tempp[:,0])
                rguh[i,0] = node[i,:]@coefficient3[1:3]/h + coefficient3[0] - center@coefficient3[1:3]/h
                coefficient4 =  np.linalg.solve(X.T@X,X.T@tempp[:,1])
                rguh[i,1] = node[i,:]@coefficient4[1:3]/h + coefficient4[0] - center@coefficient4[1:3]/h
        return rguh


    def PPR(self,uh):
        space = uh.space
        mesh = space.mesh
        GD = mesh.geo_dimension()
        rguh = space.function(dim=GD)
        
        cell = mesh.ds.cell
        node = mesh.node
        
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        coefficient = np.zeros((NN,6))
        
        isBdNodes = mesh.ds.boundary_node_flag()
        neighbor = mesh.ds.cell_to_cell()
        
        row = np.arange(NC).repeat(3)
        col = cell.flatten()
        data = np.ones(NC*3)
        t2p = csr_matrix((data, (row, col)), shape=(NC,NN)).toarray()
        p2t = t2p.T
        p2p = p2t@t2p
        
        for i in range(NN):
            #np1 = [j for (j,val) in enumerate(p2p[i,:]) if val>0]
            np1, = np.nonzero(p2p[i,:])

            npn = np1.shape[0]
            if isBdNodes[i]:
                ip = np1[~isBdNodes[np1]]
                ipn = ip.shape[0]
                if ipn == 0:
                    for k in range(npn):
                        #cp = [j for (j,val) in enumerate(p2p[np1[k],:]) if val>0]
                        cp, = np.nonzero(p2p[np1[k], :])
                        np1 = np.hstack((np1,cp))
                    np1 = np.unique(np1)
                else:
                    #cp = [j for (j,val) in enumerate(p2p[ip[0],:]) if val>0]
                    cp, = np.nonzero(p2p[ip[0],:])
                    p = np.hstack((np1,cp))
                    np2 = np.unique(p)
                    if np1.shape[0] < 6:
                        for k in range(npn):
                           # cp = [j for (j,val) in enumerate(p2p[np1[k],:]) if val>0]
                            cp, = np.nonzero(p2p[np1[k],:])
                            np1 = np.hstack((np1,cp))
                        np2 = np.unique(np1)
                    np1 = np2
            else:
                if npn < 6:
                    ne, =np.nonzero(p2t[i,:]) 
         #           ne = [j for (j,val) in enumerate(p2t[i,:]) if val>0]
                    n1 = neighbor[ne,0]
                    n2 = neighbor[ne,1]
                    n3 = neighbor[ne,2]
                    e = np.hstack((ne,n1,n2,n3))
                    e = np.unique(e)
                    p = cell[e, :]
                   # p = p[:]
                    np1 = np.unique(p)
            temp0 = node[np1,:]
            tempp = uh[np1]
            tempx,center,h = scaleCoor(temp0)
            tempn = tempx.shape[0]
            X = np.ones((tempn,6))
            X[:,1:3] = tempx
            X[:,3] = tempx[:,0]*tempx[:,1]
            X[:,4:6] = tempx**2
            cc = np.linalg.solve(X.T@X,X.T@tempp) 
            c1 = center[0]
            c2 = center[1]
            
            coefficient[i,0] = cc[0] - cc[1]*c1/h - cc[2]*c2/h +cc[3]*c1*c2/h/h
            + cc[4]*c1*c1/h/h + cc[5]*c2*c2/h/h
            coefficient[i,1] = cc[1]/h - cc[3]*c2/h/h - 2*cc[4]*c1/h/h 
            coefficient[i,2] = cc[2]/h - cc[3]*c1/h/h - 2*cc[5]*c2/h/h
            coefficient[i,3] = cc[3]/h/h
            coefficient[i,4] = cc[4]/h/h
            coefficient[i,5] = cc[5]/h/h   


        rguh[:,0] = coefficient[:,1]+coefficient[:,3]*node[:,1]+2*coefficient[:,4]*node[:,0]
        rguh[:,1] = coefficient[:,2]+coefficient[:,3]*node[:,0]+2*coefficient[:,5]*node[:,1]

        return rguh



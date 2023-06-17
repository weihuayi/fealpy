import numpy as np

class SurfaceTriangleMeshOptAlg:
    def __init__(self, surface, mesh, gamma=1, theta=0.1):
        self.surface = surface
        self.mesh=mesh
        self.gamma=gamma
        self.theta = theta

    def run(self, maxit=10):
        mesh = self.mesh
        print('Laplace smooting:')
        for i in range(0):
            mesh.node = self.laplace_smoothing()
            mesh.edge_swap()

        print('CVT smoothing:')
        for i in range(maxit):
            print(i)
            mesh.node = self.cvt_smoothing()
            isNonDelaunayEdge = mesh.edge_swap()
            return isNonDelaunayEdge

    def laplace_smoothing(self):
        mesh = self.mesh
        surface = self.surface
        theta = self.theta

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        node2node= mesh.ds.node_to_node()
        NV = np.asarray(node2node.sum(axis=1))
        newNode = np.asarray(node2node@node)/NV.reshape(-1, 1)

        normal = surface.unit_normal(node)
        mv = newNode - node
        mv -= np.sum(mv*normal, axis=-1).reshape(-1, 1)*normal
        return node + theta*mv
    
    def cvt_smoothing(self):
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        cellRho, normal= self.curvature_density()

        ec = mesh.entity_barycenter('edge')
        cell2edge = mesh.ds.cell_to_edge()

        c = self.circumcenter() 
        nodeArea = np.zeros(NN, dtype=np.float)
        massCenter = np.zeros(node.shape, dtype=np.float)
        # 0-th vertex
        a0 = np.cross(ec[cell2edge[:, 2]] - node[cell[:, 0]], c - node[cell[:, 0]])
        a1 = np.cross(c - node[cell[:, 0]], ec[cell2edge[:, 1]] - node[cell[:, 0]])
        a = 0.5*cellRho*(np.sqrt(np.sum(a0**2, axis=1)) + np.sqrt(np.sum(a1**2, axis=1)))
        p = a.reshape(-1, 1)*(node[cell[:, 0]] + c + ec[cell2edge[:, 2]] + ec[cell2edge[:, 1]])/4
        np.add.at(nodeArea, cell[:, 0], a)
        for i in range(3):
            np.add.at(massCenter[:, i], cell[:, 0], p[:, i])

        # 1-th vertex
        a0 = np.cross(ec[cell2edge[:, 0]] - node[cell[:, 1]], c - node[cell[:, 1]])
        a1 = np.cross(c - node[cell[:, 1]], ec[cell2edge[:, 2]] - node[cell[:, 1]])
        a = 0.5*cellRho*(np.sqrt(np.sum(a0**2, axis=1)) + np.sqrt(np.sum(a1**2, axis=1)))
        p = a.reshape(-1, 1)*(node[cell[:, 1]] + c + ec[cell2edge[:, 2]] + ec[cell2edge[:, 0]])/4
        np.add.at(nodeArea, cell[:, 1], a)
        for i in range(3):
            np.add.at(massCenter[:, i], cell[:, 1], p[:, i])

        # 2-th vertex
        a0 = np.cross(ec[cell2edge[:, 1]] - node[cell[:, 2]], c - node[cell[:, 2]])
        a1 = np.cross(c - node[cell[:, 2]], ec[cell2edge[:, 0]] - node[cell[:, 2]])
        a = 0.5*cellRho*(np.sqrt(np.sum(a0**2, axis=1)) + np.sqrt(np.sum(a1**2, axis=1)))
        p = a.reshape(-1, 1)*(node[cell[:, 2]] + c + ec[cell2edge[:, 1]] + ec[cell2edge[:, 0]])/4
        np.add.at(nodeArea, cell[:, 2], a)
        for i in range(3):
            np.add.at(massCenter[:, i], cell[:, 2], p[:, i])

        massCenter /= nodeArea.reshape(-1, 1)
        mv = massCenter - node
        mv -= np.sum(mv*normal, axis=-1).reshape(-1, 1)*normal

        return node + mv

    def circumcenter(self):
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        v0 = node[cell[:, 0], :]
        v1 = node[cell[:, 1], :]
        v2 = node[cell[:, 2], :]

        v10 = v1 - v0
        v20 = v2 - v0
        v21 = v2 - v1

        L10 = np.sum(v10**2, axis=1)
        L20 = np.sum(v20**2, axis=1)
        L21 = np.sum(v21**2, axis=1)

        flag0 = (L10 + L20 < L21)
        flag1 = (L21 + L10 < L20)
        flag2 = (L20 + L21 < L10)

        c, _= mesh.circumcenter()

        c[flag0] = (v1[flag0] + v2[flag0])/ 2
        c[flag1] = (v0[flag1] + v2[flag1])/ 2
        c[flag2] = (v0[flag2] + v1[flag2])/ 2

        return c

    def curvature_density(self):
        mesh = self.mesh
        surface = self.surface

        gamma = self.gamma
        theta = self.theta

        NN = mesh.number_of_nodes()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        
        normal = surface.unit_normal(node)

        n = np.sum(normal[cell], axis=1) 
        v01 = node[cell[:,1],:] - node[cell[:,0],:]
        v02 = node[cell[:,2],:] - node[cell[:,0],:]
        
        nv = np.cross(v01,v02)
        length = np.sqrt(np.sum(nv**2, axis=1))
        nv = nv/length.reshape(-1, 1)
        area = length/2

        cellRho = np.abs(9-np.sum(n**2, axis=1))/area
        cellRho = cellRho + theta*np.max(cellRho) 
        sumArea = np.bincount(cell.flat, weights=np.repeat(area, 3), minlength=NN)

        for i in range(12):
            nodeRho = np.bincount(cell.flat, weights=np.repeat(cellRho*area, 3), minlength=NN)
            nodeRho /= sumArea
            cellRho = np.sum(nodeRho[cell], axis=-1)/3 

        cellRho = (cellRho/np.max(cellRho))**(gamma)
        return cellRho, normal







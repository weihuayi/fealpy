import numpy as np
from .simple_mesh_generator import rectangledomainmesh  
from .simple_mesh_generator import triangle, unitsquaredomainmesh

class MeshZoo():
    #Fishbone
    def regular(self, box, n=10):
        return rectangledomainmesh(box, nx=n, ny=n, meshtype='tri')

    def fishbone(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NC = qmesh.number_of_cells()
        isLeftCell = np.zeros((n, n), dtype=np.bool)
        isLeftCell[0::2, :] = True
        isLeftCell = isLeftCell.reshape(-1)
        lcell = cell[isLeftCell]
        rcell = cell[~isLeftCell]
        newCell = np.r_['0', 
                lcell[:, [1, 2, 0]], 
                lcell[:, [3, 0, 2]],
                rcell[:, [0, 1, 3]],
                rcell[:, [2, 3, 1]]]
        return TriangleMesh(node, newCell)

        #cross mesh
    def cross_mesh(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NN = qmesh.number_of_nodes()
        NC = qmesh.number_of_cells()
        bc = qmesh.barycenter('cell') 
        newNode = np.r_['0', node, bc]

        newCell = np.zeros((4*NC, 3), dtype=np.int) 
        newCell[0:NC, 0] = range(NN, NN+NC)
        newCell[0:NC, 1:3] = cell[:, 0:2]
        
        newCell[NC:2*NC, 0] = range(NN, NN+NC)
        newCell[NC:2*NC, 1:3] = cell[:, 1:3]

        newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
        newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

        newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
        newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
        return TriangleMesh(newNode, newCell)

    def rice_mesh(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NC = qmesh.number_of_cells()

        isLeftCell = np.zeros((n, n), dtype=np.bool)
        isLeftCell[0, 0::2] = True
        isLeftCell[1, 1::2] = True
        if n > 2:
            isLeftCell[2::2, :] = isLeftCell[0, :]
        if n > 3:
            isLeftCell[3::2, :] = isLeftCell[1, :]
        isLeftCell = isLeftCell.reshape(-1)

        lcell = cell[isLeftCell]
        rcell = cell[~isLeftCell]
        newCell = np.r_['0', 
                lcell[:, [1, 2, 0]], 
                lcell[:, [3, 0, 2]],
                rcell[:, [0, 1, 3]],
                rcell[:, [2, 3, 1]]]
        return TriangleMesh(node, newCell)

    def delaunay_mesh(self, box, n=10):
        h = (box[1] - box[0])/n
        return triangle(box, h, meshtype='tri')

    def nonuniform_mesh(self, box, n=10):
        nx = 4*n
        ny = 4*n
        n1 = 2*n+1

        N = n1**2
        NC = 4*n*n
        node = np.zeros((N,2))
        
        x = np.zeros(n1, dtype=np.float)
        x[0::2] = range(0,nx+1,4)
        x[1::2] = range(3, nx+1, 4)

        y = np.zeros(n1, dtype=np.float)
        y[0::2] = range(0,nx+1,4)
        y[1::2] = range(1, nx+1,4)

        node[:,0] = x.repeat(n1)/nx
        node[:,1] = np.tile(y, n1)/ny


        idx = np.arange(N).reshape(n1, n1)
        
        cell = np.zeros((2*NC, 3), dtype=np.int)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
        return TriangleMesh(node, cell)
        
    def uncross_mesh(self, box, n=10, r="1"):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NN = qmesh.number_of_nodes()
        NC = qmesh.number_of_cells()   
        bc = qmesh.barycenter('cell') 

        if r=="1":
            bc1 = np.sqrt(np.sum((bc-node[cell[:,0], :])**2, axis=1))[0]
            newNode = np.r_['0', node, bc-bc1*0.3]
        elif r=="2":
            ll = node[cell[:, 0]] - node[cell[:, 2]]
            bc = qmesh.barycenter('cell') + ll/4
            newNode = np.r_['0',node, bc]

        newCell = np.zeros((4*NC, 3), dtype=np.int) 
        newCell[0:NC, 0] = range(NN, NN+NC)
        newCell[0:NC, 1:3] = cell[:, 0:2]
            
        newCell[NC:2*NC, 0] = range(NN, NN+NC)
        newCell[NC:2*NC, 1:3] = cell[:, 1:3]

        newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
        newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

        newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
        newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
        return TriangleMesh(newNode, newCell)

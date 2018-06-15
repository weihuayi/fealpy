import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

class TriRadiusRatioQuality():
    def __init__(self, mesh):
        self.mesh = mesh;

    def __call__(self, x):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node.copy()
        isBdNode = self.mesh.ds.boundary_node_flag()
        NB = isBdNode.sum()
        node[~isBdNode] = x.reshape(NN - NB, 2)
        mu = self.quality(node)

        f = np.mean(mu)
        return f

    def get_init_value(self):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node.copy()
        isBdNode = self.mesh.ds.boundary_node_flag()
        x0 = node[~isBdNode].reshape(-1)
        return x0


    def callback(self, x):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node.copy()
        isBdNode = self.mesh.ds.boundary_node_flag()
        NB = isBdNode.sum()
        node[~isBdNode] = x.reshape(NN - NB, 2)
        flag = self.is_valid(node)
        q = self.quality(node)
        print('Max quality:', q.max())
        print('All area > 0:', flag)
        return False 


    def update_mesh_node(self, x):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node
        isBdNode = self.mesh.ds.boundary_node_flag()
        NB = isBdNode.sum()
        node[~isBdNode] = x.reshape(NN - NB, 2)

    def quality(self, node=None):
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells() 
        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2 
        quality = p*q/(16*area**2)
        return quality

    def quality_with_gradient(self, node=None):
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        
        NN = self.mesh.number_of_nodes() 
        NC = self.mesh.number_of_cells() 

        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2 
        mu = p*q/(16*area**2)

        c = mu[:, None]*(1/(p[:, None]*l) + 1/l2)
        val = np.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = np.einsum('ij, k->ijk', cell, np.ones(3))
        J = I.swapaxes(-1, -2)
        A = csr_matrix((val, (I.flat, J.flat)), shape=(NN, NN))

        cn = mu/area;
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((
            cell[:, 0], cell[:, 0], 
            cell[:, 1], cell[:, 1],
            cell[:, 2], cell[:, 2]))
        J = np.concatenate((
            cell[:, 1], cell[:, 2], 
            cell[:, 0], cell[:, 2],
            cell[:, 0], cell[:, 1]))
        B = csr_matrix((val, (I, J)), shape=(NN, NN))

        grad = np.zeros((NN, 2), dtype=np.float)
        grad[:, 0] = A@node[:, 0] + B@node[:, 1]
        grad[:, 1] = -B@node[:, 0] + A@node[:, 1]
        return  mu, -grad

    def is_valid(self, node=None):
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells() 
        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        area = np.cross(v[2], -v[1])/2 
        return np.all(area > 0)

import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import csr_matrix

from fealpy.mesh import TriangleMesh
from fealpy.opt import Problem, MatrixVectorProductGradientOptimizer

import ipdb

class TriMeshProblem(Problem):
    def __init__(self,mesh:TriangleMesh):
        self.mesh = mesh
        node = mesh.entity('node')
        self.isBdNode = mesh.ds.boundary_node_flag()
        x0 = np.array(node[~self.isBdNode, :].T.flat)

        super().__init__(x0, self.quality)

    def quality(self, x):
        GD = self.mesh.geo_dimension()
        node0 = self.mesh.entity('node')
        node = np.full_like(node0, 0.0)

        node[self.isBdNode, :] = node0[self.isBdNode, :]
        NI = np.sum(~self.isBdNode)
        node[~self.isBdNode,0] = x[:NI]
        node[~self.isBdNode,1] = x[NI:]

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

        A, B = self.grad_matrix(node=node)

        gradp = np.full_like(x, 0.0).reshape(GD, -1)
        gradp[0, :] = (A@node[:, 0] + B@node[:, 1])[~self.isBdNode]
        gradp[1, :] = (B.T@node[:, 0] + A@node[:, 1])[~self.isBdNode]
        return np.mean(quality), gradp.flat

    def grad_matrix(self, node=None):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')


        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2] 

        v0 = node[idxk] - node[idxj]
        v1 = node[idxi] - node[idxk]
        v2 = node[idxj] - node[idxi]
        area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
        l2 = np.zeros((NC, 3), dtype=np.float64)
        l2[:, 0] = np.sum(v0**2, axis=1)
        l2[:, 1] = np.sum(v1**2, axis=1)
        l2[:, 2] = np.sum(v2**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        mu = p*q/(16*area**2)
        c = mu*(1/(p*l) + 1/l2)
        val = np.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = np.concatenate((
            idxi, idxi, idxi,
            idxj, idxj, idxj,
            idxk, idxk, idxk))
        J = np.concatenate((idxi, idxj, idxk))
        J = np.concatenate((J, J, J))
        A = csr_matrix((val, (I, J)), shape=(NN, NN))

        cn = mu/area
        cn.shape = (cn.shape[0],)
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((idxi, idxi, idxj, idxj, idxk, idxk))
        J = np.concatenate((idxj, idxk, idxi, idxk, idxi, idxj))
        B = csr_matrix((val, (I, J)), shape=(NN, NN))
        return (A, B)


    def block_jacobi_preconditioner(self, x):
        isBdNode = self.isBdNode
        isFreeNode = ~isBdNode

        node0 = self.mesh.entity('node')
        node = np.full_like(node0, 0.0)

        node[isBdNode, :] = node0[isBdNode, :]
        #node[isFreeNode, :].T.flat[:] = x
        NI = np.sum(isFreeNode)
        node[isFreeNode,0] = x[:NI]
        node[isFreeNode,1] = x[NI:]

        A, B = self.grad_matrix(node=node)

        node1 = np.full_like(node, 0.0)
        node1[isBdNode, :] = node[isBdNode, :]        

        b = -B*node[:, 1] - A*node1[:, 0]
        node1[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)], b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
        b = B*node[:, 0] - A*node1[:, 1]
        node1[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)], b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6)

        return np.array((node1[isFreeNode, :] - node[isFreeNode, :]).T.flat)

def test_triangle_mesh_opt():
    mesh = TriangleMesh.from_unit_circle_gmsh(h=0.1)
    #mesh = TriangleMesh.from_one_triangle('equ')
    #mesh.uniform_refine(n=3)
    area = mesh.entity_measure('cell')
    problem = TriMeshProblem(mesh)

    NDof = len(problem.x0)
    problem.Preconditioner = LinearOperator((NDof, NDof), problem.block_jacobi_preconditioner, dtype=np.float64)
    problem.StepLength = 1.0
    opt = MatrixVectorProductGradientOptimizer(problem)
    x, f, g = opt.run()

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)

    node = mesh.entity('node')
    isFreeNode = ~mesh.ds.boundary_node_flag()
    n = len(x)//2 
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:]

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)

    plt.show()

if __name__ == "__main__":
    test_triangle_mesh_opt()

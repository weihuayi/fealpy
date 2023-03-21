import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix, spdiags, triu, tril, find, hstack, eye, bmat
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from scipy.linalg import norm
from pyamg import *

class TetRadiusRatio():
    def __init__(self,mesh):
        self.mesh = mesh

    def get_free_node_info(self):
        NN = self.mesh.number_of_nodes()
        isBdNode = self.mesh.ds.boundary_node_flag()
        isFreeNode = np.ones((NN, ), dtype=np.bool)
        isFreeNode[isBdNode] = False
        return isFreeNode

    def get_quality(self):
        mesh = self.mesh
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells() 

        vji = node[cell[:, 0]] - node[cell[:, 1]] 
        vki = node[cell[:, 0]] - node[cell[:, 2]] 
        vmi = node[cell[:, 0]] - node[cell[:, 3]] 

        vji2 = np.sum(vji**2, axis=-1)
        vki2 = np.sum(vki**2, axis=-1)
        vmi2 = np.sum(vmi**2, axis=-1)

        d = vmi2[:, None]*(np.cross(vji, vki)) + vji2[:, None]*(np.cross(vki, vmi)
                ) + vki2[:, None]*(np.cross(vmi, vji))
        dl = np.sqrt(np.sum(d**2, axis=-1))

        fm = mesh.entity_measure("face")
        cm = mesh.entity_measure("cell")
        c2f = mesh.ds.cell_to_face()

        s_sum = np.sum(fm[c2f], axis=-1)
        mu = s_sum*dl/108/cm/cm
        return mu

    def get_iterate_matrix(self):
        pass

    def grad(self):
        mesh = self.mesh
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells() 

        vji = node[cell[:, 0]] - node[cell[:, 1]] 
        vki = node[cell[:, 0]] - node[cell[:, 2]] 
        vmi = node[cell[:, 0]] - node[cell[:, 3]] 
        vjm = node[cell[:, 3]] - node[cell[:, 1]]
        vmk = node[cell[:, 2]] - node[cell[:, 3]]
        vkj = node[cell[:, 1]] - node[cell[:, 2]]

        vji2 = np.sum(vji**2, axis=-1)
        vki2 = np.sum(vki**2, axis=-1)
        vmi2 = np.sum(vmi**2, axis=-1)
        vjm2 = np.sum(vjm**2, axis=-1)
        vmk2 = np.sum(vmk**2, axis=-1)
        vkj2 = np.sum(vkj**2, axis=-1)

        d = vmi2[:, None]*(np.cross(vji, vki)) + vji2[:, None]*(np.cross(vki, vmi)
                ) + vki2[:, None]*(np.cross(vmi, vji))

        dl = np.sqrt(np.sum(d**2, axis=-1))
        ckm = np.sum(d*np.cross(vki, vmi), axis=-1) 
        cmj = np.sum(d*np.cross(vmi, vji), axis=-1) 
        cjk = np.sum(d*np.cross(vji, vki), axis=-1) 

        A  = np.zeros((NC, 4, 4), dtype=np.float_)
        K = np.zeros((NC, 4, 4), dtype=np.float_)

        A[:, 0, 1] = -2*ckm
        A[:, 0, 2] = -2*cmj
        A[:, 0, 3] = -2*cjk
        A[:, 0, 0] = 2*(ckm+cmj+cjk)
        A[:, 1, 1] = 2*ckm
        A[:, 2, 2] = 2*cmj
        A[:, 3, 3] = 2*cjk
        A[:, :, 0] = A[:, 0, :]

        K[:, 1, 0] = vmi2-vki2
        K[:, 2, 0] = vji2-vmi2
        K[:, 3, 0] = vki2-vji2
        K[:, 2, 1] = vmi2
        K[:, 3, 1] = -vki2
        K[:, 3, 2] = vji2
        K[:, 0, :] = -K[:, :, 0]
        K[:, 1, 1:] = -K[:, 1:, 1]
        K[:, 2, 3] = -vji2

        A /= dl**2
        K /= dl**2
        
        B1 = -d[:, 2, None, None]*K
        C  =  d[:, 1, None, None]*K
        B2 = -d[:, 0, None, None]*K

        fm = mesh.entity_measure("face")
        cm = mesh.entity_measure("cell")
        c2f = mesh.ds.cell_to_face()

        ## \nabla s_sum
        s = fm[c2f]
        s_sum = np.sum(s, axis=-1)
        pi = (vjm2/s[2] + vkj2/s[3] + vmk2/s[1])/4
        pj = (vmk2/s[0] + vki2/s[3] + vmi2/s[2])/4
        pk = (vmi2/s[1] + vji2/s[3] + vjm2/s[0])/4
        pm = (vji2/s[2] + vki2/s[1] + vkj2/s[0])/4

        qij = -(np.sum(-vjm*vmi, axis=-1)/s[2]+np.sum(vkj*vki, axis=-1)/s[3])/4
        qik = -(np.sum(vmk*vmi, axis=-1)/s[1]+np.sum(-vkj*vji, axis=-1)/s[3])/4
        qim = -(np.sum(-vmk*vki, axis=-1)/s[1]+np.sum(vjm*vji, axis=-1)/s[2])/4
        qjk = -(np.sum(-vmk*vjm, axis=-1)/s[0]+np.sum(vki*vji, axis=-1)/s[3])/4
        qjm = -(np.sum(vmi*vji, axis=-1)/s[2]+np.sum(-vmk*vkj, axis=-1)/s[0])/4
        qkm = -(np.sum(-vjm*vkj, axis=-1)/s[0]+np.sum(vmi*vki, axis=-1)/s[1])/4

        As = np.zeros_like(A)
        As[:, 0, 0] = pi
        As[:, 0, 1] = qij
        As[:, 0, 2] = qik
        As[:, 0, 3] = qim

        As[:, 1, 1] = pj
        As[:, 1, 2] = qjk
        As[:, 1, 3] = qjm

        As[:, 2, 2] = pk
        As[:, 2, 3] = qkm

        As[:, 3, 3] = pm

        As[:, :, 0] = As[:, 0, :]
        As[:, :, 1] = As[:, 1, :]
        As[:, :, 2] = As[:, 2, :]

        As /= s_sum

        A += As

        ## \nabla |\tau|
        C0 = np.zeros((NC, 4, 4), dtype=np.float_)
        C1 = np.zeros((NC, 4, 4), dtype=np.float_)
        C2 = np.zeros((NC, 4, 4), dtype=np.float_)
        def f(CC, xx):
            CC[:, 0, 1] = xx[:, 2]
            CC[:, 0, 2] = xx[:, 3]
            CC[:, 0, 3] = xx[:, 1]
            CC[:, 1, 0] = xx[:, 3]
            CC[:, 1, 2] = xx[:, 0]
            CC[:, 1, 3] = xx[:, 2]
            CC[:, 2, 0] = xx[:, 1]
            CC[:, 2, 1] = xx[:, 3]
            CC[:, 2, 3] = xx[:, 0]
            CC[:, 3, 0] = xx[:, 2]
            CC[:, 3, 1] = xx[:, 0]
            CC[:, 3, 2] = xx[:, 1]

        f(C0, node[cell, 0])
        f(C1, node[cell, 1])
        f(C2, node[cell, 2])
        C0 = 0.5*(-C0 + C0.swapaxes(-1, -2))
        C1 = 0.5*(C1  - C1.swapaxes(-1, -2))
        C2 = 0.5*(-C2 + C2.swapaxes(-1, -2))

        B1 += -C2/cm[:, None, None]/3
        B2 += -C0/cm[:, None, None]/3
        C  += -C1/cm[:, None, None]/3

        mu = s_sum*dl/108/cm/cm
        A *= mu/NC
        B1 *= mu/NC
        B2 *= mu/NC
        C *= mu/NC

        I = np.broadcast_to(cell[:, :, None], (NC, 4, 4))
        J = np.broadcast_to(cell[:, None, :], (NC, 4, 4))
        A  = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN))
        B1 = csr_matrix((B1.flat, (I.flat, J.flat)), shape=(NN, NN))
        B2 = csr_matrix((B2.flat, (I.flat, J.flat)), shape=(NN, NN))
        C  = csr_matrix((C.flat, (I.flat, J.flat)), shape=(NN, NN))
        print("B1", np.sum(B1+B1.T))
        print('B2', np.sum(B2+B2.T))
        print('C', np.sum(C+C.T))
        return  A, B1, B2, C


    def iterate_solver(self, method='jacobi'):
        NC = self.mesh.number_of_cells()
        node = self.mesh.entity('node')
        isFreeNode = self.get_free_node_info()        
        q = np.zeros((2, NC), dtype=np.float_)
        if method=='jacobi':
            for i in range(0,500):
                A, B, D, C = self.grad()
                node1=self.Jacobi(node, A, B, C, D, isFreeNode)		
                            ## count quality
                q[1] = self.get_quality()
                minq = np.min(q)
                avgq = np.mean(q)
                if np.max(np.abs(q[1]-q[0]))<1e-8:
                    print("jacobi迭代次数为%d次"%(i+1))
                    break
                q[0] = q[1]
                node=node1
            print("jacobi迭代次数为500次")
            
        if method=='Bjacobi':
            for i in range(0,500):
                A, B, D, C = self.grad()
                node1 = self.BlockJacobi(node, A, B, C, D, isFreeNode)
                            ## count quality
                q[1] = self.get_quality()
                minq = np.min(q)
                avgq = np.mean(q)
                #print('minq=',minq,'avgq=',avgq)
                if np.max(np.abs(q[1]-q[0]))<1e-8:
                    print("Bjacobi迭代次数为%d次"%(i+1))
                    break
                q[0] = q[1]
                node[:]=node1
        if method=='BGauss':
            for i in range(0,500):
                A, B, D, C = self.grad()
                node1 = self.BlockGauss(node, A, B, C, D, isFreeNode)
                            ## count quality
                q[1] = self.get_quality()
                minq = np.min(q)
                avgq = np.mean(q)
                #print('minq=',minq,'avgq=',avgq)
                if np.max(np.abs(q[1]-q[0]))<1e-8:
                    print("BGauss迭代次数为%d次"%(i+1))
                    break
                q[0] = q[1]
                node=node1
        if method=='cg':
            for i in range(0,5):
                A, B, D, C = self.grad()

                A = bmat([[A, B, C], [B.T, A, D], [C.T, D.T, A]])

                node111 = self.mesh.entity('node').copy()

                bbb = node111.T.flatten()
                print("aaaaaa", A@bbb)

                node111[isFreeNode] = 0

                b = node111.T.flatten()

                #F = -A@b

                flag = np.r_[~isFreeNode, ~isFreeNode, ~isFreeNode]
                bdIdx = np.zeros(A.shape[0], dtype=np.int_)
                bdIdx[flag] = 1

                Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
                T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
                A = T@A + Tbd
                #F[flag] = b[flag]

                #val, _ = cg(A, F, tol=1e-6)
                val = spsolve(A, b)
                node[:] = val.reshape(3, -1).T
        return node
    def Jacobi(self, node, A, B , C, D, isFreeNode):
        NN = self.mesh.number_of_nodes() 
        D = spdiags(1.0/A.diagonal(), 0, NN, NN)
        M = -(triu(A, 1) + tril(A, -1))
        X = D*(M*node[:, 0] - B*node[:, 1] - C*node[:,2])
        Y = D*(B*node[:, 0] + M*node[:, 1] - D*node[:,2])
        Z = D*(C*node[:, 0] + D*node[:, 1] + M*node[:,2])
        p = np.zeros((NN, 3)) 
        p[isFreeNode, 0] = X[isFreeNode] - node[isFreeNode, 0]
        p[isFreeNode, 1] = Y[isFreeNode] - node[isFreeNode, 1]
        p[isFreeNode, 2] = Z[isFreeNode] - node[isFreeNode, 2]
        node +=100*p/NN
        #node[isFreeNode,0]=X[isFreeNode]
        #node[isFreeNode,1]=Y[isFreeNode]
        return node


    def BlockJacobi(self, node, A, B, C, D, isFreeNode):
        NN = self.mesh.number_of_nodes() 
        isBdNode = np.logical_not(isFreeNode)
        newNode = np.zeros((NN, 3), dtype=np.float)
        newNode[isBdNode, :] = node[isBdNode, :]        
        b = -B*node[:, 1] -C*node[:, 2] - A*newNode[:, 0]
        newNode[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
        b = B*node[:, 0] - A*newNode[:, 1] - D*node[:, 2]
        newNode[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6)
        b = C*node[:,0]+D*node[:,1]-A*newNode[:,2]
        newNode[isFreeNode, 2], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 2], tol=1e-6)
        node[isFreeNode, :] = newNode[isFreeNode, :]
        return node

    def BlockGauss(self, node, A, B, C, D, isFreeNode):
        NN = self.mesh.number_of_nodes() 
        isBdNode = np.logical_not(isFreeNode)
        newNode = np.zeros((NN, 3), dtype=np.float)

        newNode[isBdNode, :] = node[np.ix_(isBdNode, [0, 1, 2])]
        ml = smoothed_aggregation_solver(A[np.ix_(isFreeNode, isFreeNode)])
        M = ml.aspreconditioner(cycle='W')
        b = -B*node[:, 1] - -C*node[:, 2] - A*newNode[:, 0]
        node[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6, M=M)
        b = B*node[:, 0] - A*newNode[:, 1] - D*node[:, 2]
        node[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6, M=M)
        b = C*node[:, 0] + D*node[:, 1] - A*newNode[:, 2]
        node[isFreeNode, 2], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 2], tol=1e-6, M=M)
        return node


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
from .TriangleMesh import TriangleMesh

from scipy.sparse import csc_matrix, csr_matrix, spdiags, triu, tril, find, hstack, eye
from scipy.sparse.linalg import cg, inv, dsolve


class OptMesh2d():
    def __init__(self, domain, trimesh, dptol = 0.001, ttol = 0.1):

        self.domain = domain
        self.dptol = dtptol
        self.ttol = ttol

        eps = np.finfo(float).eps
        self.geps = 0.001*h0
        self.deps = np.sqrt(eps)*h0
        self.dt = 0.2

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.mesh = trimesh 
        self.mesh.auxstructure()
        edge = self.mesh.T.edge
        edge2cell = self.mesh.T.edge2cell
        N = self.mesh.number

    def run(self):
        while True: 
            dt = self.step_length()
            self.step(dt)
            if  self.maxmove < self.dptol:
                break

    def step_length(self):
        return self.dt

    def step(self, dt):
        fd, fh, bbox, pfix, args = self.domain.params
        dxdt = self.dx_dt(self.time_elapsed)
        self.mesh.point = self.mesh.point + dt*dxdt

        p = self.mesh.point
        d = fd(p, *args)
        idx = d > 0
        depsx = np.array([self.deps,0])
        depsy = np.array([0,self.deps])
        dgradx = (fd(p[idx, :]+depsx, *args) - d[idx])/self.deps
        dgrady = (fd(p[idx, :]+depsy, *args) - d[idx])/self.deps
        p[idx, 0] = p[idx, 0] - d[idx]*dgradx
        p[idx, 1] = p[idx, 1] - d[idx]*dgrady

        self.maxmove = np.max(np.sqrt(np.sum(dt*dxdt[d < -self.geps,:]**2, axis=1))/h0)
        self.time_elapsed += dt
        if self.maxmove > ttol:
            self.mesh.cell = self.delaunay(self.mesh.point)

    def dx_dt(self, t):

        fd, fh, bbox, pfix, args = self.domain.params
        dptol, ttol = self.params
        point = self.mesh.point
        N = point.shape[0]

        A, B = self.get_iterate_matrix()
        D = spdiags(1.0/A.diagonal(), 0, N, N)
        C = -(triu(A, 1) + tril(A, -1))
        X = D*(C*point[:, 0] - B*point[:, 1])
        Y = D*(B*point[:, 0] + C*point[:, 1])

        dxdt = np.zeros((N, 2), dtype=np.float)
        dxdt[:,0] = X - point[:,0]
        dxdt[:,1] = Y - point[:,1]
       
        if pfix is not None:
            dxdt[0:pfix.shape[0],:] = 0
        return dxdt 

    def delaunay(self, p):
        fd, *_, args = self.domain.params
        d = Delaunay(p)
        t = d.simplices
        pc = (p[t[:, 0], :]+p[t[:, 1], :]+p[t[:, 2], :])/3
        return  t[fd(pc, *args) < - self.geps, :]

    def get_iterate_matrix(self):
        point = self.mesh.point
        cell = self.mesh.cell

        N = point.shape[0]
        NC = cell.shape[0]
        idxi = cell[:,0]
        idxj = cell[:,1]
        idxk = cell[:,2]

        v0 = point[idxk,:] - point[idxj,:]
        v1 = point[idxi,:] - point[idxk,:]
        v2 = point[idxj,:] - point[idxi,:]

        area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
        l2 = np.zeros((NC, 3), dtype=np.float)
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
        A = csr_matrix((val, (I, J)), shape=(N, N))

        cn = mu/area
        cn.shape = (cn.shape[0],)
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((idxi, idxi, idxj, idxj, idxk, idxk))
        J = np.concatenate((idxj, idxk, idxi, idxk, idxi, idxj))
        B = csr_matrix((val, (I, J)), shape=(N, N))

        return (A, B)


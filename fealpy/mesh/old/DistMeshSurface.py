import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from skimage import measure
from .TriangleMesh import TriangleMesh

class DistMeshSurface():
    def __init__(self,
            domain, 
            h0,
            dptol = 0.0001,
            ttol = 0.1,
            Fscale = 1.2):

        self.domain = domain
        self.params = (h0, dptol, ttol, Fscale)

        eps = np.finfo(float).eps
        self.deps = np.sqrt(eps)*h0
        self.dt = 0.2

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.set_init_mesh()

    def run(self):
        h0, dptol, ttol, Fscale = self.params
        while True: 
            dt = self.step_length()
            self.step(dt)
            if  self.maxmove < dptol:
                break

    def set_init_mesh(self): 

        fd, fh, bbox, pfix, args = self.domain.params
        h0, dptol, ttol, Fscale = self.params

        xh = bbox[1] - bbox[0]
        yh = bbox[3] - bbox[2]
        zh = bbox[5] - bbox[4]
        N = int(xh/h0)+1
        M = int(yh/h0)+1
        K = int(zh/h0)+1

        X, Y, Z = np.mgrid[
                bbox[0]:bbox[1]:complex(0, M),
                bbox[2]:bbox[3]:complex(0, N),
                bbox[4]:bbox[5]:complex(0, K)]
        p = np.concatenate((
            X.reshape(-1,1),
            Y.reshape(-1,1),
            Z.reshape(-1,1)), axis=1)
        surf_eq = fd(p,*args)
        p, t = measure.marching_cubes(surf_eq.reshape(N,M,K), 0)
        p = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)
        p = p[fd(p, *args) < self.geps, :]
        r0 = 1/fh(p, *args)**2
        p = p[np.random.random((p.shape[0],)) < r0/np.max(r0),:]
        if pfix is not None:
            p = np.concatenate((pfix, p), axis=0)

        t = self.delaunay(p)
        self.mesh = TriangleMesh(p, t)
        self.edge = self.mesh.ds.edge

    def step_length(self):
        return self.dt

    def step(self, dt):
        fd, fh, bbox, pfix, args = self.domain.params
        h0, dptol, ttol, Fscale = self.params

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
            t = self.delaunay(self.mesh.point)
            self.mesh = TriangleMesh(self.mesh.point, t)
            self.edge = self.mesh.ds.edge

    def dx_dt(self, t):

        fd, fh, bbox, pfix, args = self.domain.params
        h0, dptol, ttol, Fscale = self.params

        p = self.mesh.point
        N = p.shape[0]
        edge = self.edge
        vec = p[edge[:, 0], :] - p[edge[:, 1], :]
        L = np.sqrt(np.sum(vec**2, axis=1))
        hedge = fh(p[edge[:, 1],:]+vec/2, *args) 
        L0 = np.sqrt(np.sum(L**2)/np.sum(hedge**2))*Fscale*hedge
        F = L0 - L
        F[L0-L<0] = 0
        FV = (F/L).reshape((-1,1))*vec

        dxdt = np.zeros((N, 2), dtype=np.float)
        dxdt[:, 0] += np.bincount(edge[:,0], weights=FV[:,0], minlength=N)
        dxdt[:, 1] += np.bincount(edge[:,0], weights=FV[:,1], minlength=N)
        dxdt[:, 0] -= np.bincount(edge[:,1], weights=FV[:,0], minlength=N)
        dxdt[:, 1] -= np.bincount(edge[:,1], weights=FV[:,1], minlength=N)

        if pfix is not None:
            dxdt[0:pfix.shape[0],:] = 0
        return dxdt 

    def delaunay(self, p):
        fd, *_, args = self.domain.params
        d = Delaunay(p)
        t = d.simplices
        pc = (p[t[:, 0], :]+p[t[:, 1], :]+p[t[:, 2], :])/3
        return  t[fd(pc, *args) < - self.geps, :]


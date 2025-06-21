import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
from .TriangleMesh import TriangleMesh
from .TetrahedronMesh import TetrahedronMesh

class DistMesh2d():
    def __init__(self,
            domain, 
            h,
            dptol = 0.001,
            ttol = 0.1,
            Fscale = 1.2):

        self.domain = domain
        self.params = (h, dptol, ttol, Fscale)

        eps = np.finfo(float).eps
        self.geps = 0.001*h
        self.deps = np.sqrt(eps)*h
        self.dt = 0.2

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.count = 0

    def run(self, maxit=1000):
        """
        @brief 运行
        """
        dptol = self.params[1]
        self.set_init_mesh()
        count = 0
        while count < maxit: 
            dt = self.step_length()
            self.step(dt)
            count += 1
            if self.maxmove < dptol:
                break

    def show_animation(self, plot=None, axes=None, fname='test.mp4',
            fargs=None, frames=1000,  interval=50, 
            edgecolor='k', linewidths=1, aspect='equal', showaxis=False):
        import matplotlib.animation as animation
        from matplotlib.collections import LineCollection

        if plot is None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots()
        else:
            if isinstance(plot, ModuleType):
                fig = plot.figure()
                fig.set_facecolor('white')
            else:
                fig = plot
                fig.set_facecolor('white')

            if axes is None:
                axes = fig.gca()

        try:
            axes.set_aspect(aspect)
        except NotImplementedError:
            pass

        if showaxis == False:
            axes.set_axis_off()
        else:
            axes.set_axis_on()

        dptol = self.params[1]
        lines = LineCollection([], linewidths=linewidths, color=edgecolor)

        def init_func():
            self.set_init_mesh()
            node = self.mesh.entity('node')
            box = np.zeros(4, dtype=np.float64)
            box[0::2] = np.min(node, axis=0)
            box[1::2] = np.max(node, axis=0)

            tol = np.max(self.mesh.entity_measure('edge'))
            axes.set_xlim([box[0]-tol, box[1]+0.01]+tol)
            axes.set_ylim([box[2]-tol, box[3]+0.01]+tol)

            edge = self.mesh.entity('edge')
            lines.set_segments(node[edge])
            axes.add_collection(lines)

            return lines

        def func(n, *fargs):
            dt = self.step_length()
            self.step(dt)

            node = self.mesh.entity('node')
            edge = self.mesh.entity('edge')
            lines.set_segments(node[edge])
            s = "step=%05d"%(n)
            print(s)
            axes.set_title(s)
            return lines

        ani = animation.FuncAnimation(fig, func, frames=frames,
                init_func=init_func,
                interval=interval)
        ani.save(fname)

    def set_init_mesh(self): 
        """
        @brief 生成初始网格
        """

        fd, fh, bbox, pfix, args = self.domain.params
        h, dptol, ttol, Fscale = self.params

        xh = bbox[1] - bbox[0]
        yh = bbox[3] - bbox[2]
        N = int(xh/h)+1
        M = int(yh/(h*np.sqrt(3)/2))+1

        mg = np.mgrid[bbox[2]:bbox[3]:complex(0, M), bbox[0]:bbox[1]:complex(0, N)]
        x = mg[1, :, :]
        y = mg[0, :, :]
        x[1::2, :] = x[1::2, :] + h/2 
        p = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)
        p = p[fd(p, *args) < -self.geps, :]
        r0 = 1/fh(p, *args)**2
        p = p[np.random.random((p.shape[0],)) < r0/np.max(r0),:]
        if pfix is not None:
            p = np.concatenate((pfix, p), axis=0)

        t = self.delaunay(p)
        self.mesh = TriangleMesh(p, t)

    def step_length(self):
        return self.dt

    def step(self, dt):
        fd, fh, bbox, pfix, args = self.domain.params
        h, dptol, ttol, Fscale = self.params

        dxdt = self.dx_dt(self.time_elapsed)
        self.mesh.node = self.mesh.node + dt*dxdt

        p = self.mesh.node
        d = fd(p, *args)
        idx = d > 0
        depsx = np.array([self.deps, 0])
        depsy = np.array([0, self.deps])
        dgradx = (fd(p[idx, :]+depsx, *args) - d[idx])/self.deps
        dgrady = (fd(p[idx, :]+depsy, *args) - d[idx])/self.deps
        p[idx, 0] = p[idx, 0] - d[idx]*dgradx
        p[idx, 1] = p[idx, 1] - d[idx]*dgrady
        self.maxmove = np.max(np.sqrt(np.sum(dt*dxdt[d < -self.geps,:]**2, axis=1))/h)
        self.time_elapsed += dt

        if self.maxmove > ttol:
            t = self.delaunay(self.mesh.node)
            self.mesh = TriangleMesh(self.mesh.node, t)



    def dx_dt(self, t):

        fd, fh, bbox, pfix, args = self.domain.params
        h, dptol, ttol, Fscale = self.params

        p = self.mesh.node
        N = p.shape[0]
        edge = self.mesh.ds.edge
        vec = p[edge[:, 0], :] - p[edge[:, 1], :]
        L = np.sqrt(np.sum(vec**2, axis=1))
        hedge = fh(p[edge[:, 1],:]+vec/2, *args) 
        L0 = np.sqrt(np.sum(L**2)/np.sum(hedge**2))*Fscale*hedge
        F = L0 - L
        F[L0-L<0] = 0
        FV = (F/L).reshape((-1,1))*vec

        dxdt = np.zeros((N, 2), dtype=np.float64)
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
        t = np.asarray(d.simplices, dtype=np.int_)
        pc = (p[t[:, 0], :]+p[t[:, 1], :]+p[t[:, 2], :])/3
        return  t[fd(pc, *args) < -self.geps, :]

class DistMesh3d:
    def __init__(self,
            domain, 
            h,
            dptol = 0.001,
            ttol = 0.1,
            Fscale = 1.1):

        self.domain = domain
        self.params = (h, dptol, ttol, Fscale)

        eps = np.finfo(float).eps
        self.geps = 0.1*h
        self.deps = np.sqrt(eps)*h
        self.dt = 0.1

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.set_init_mesh()

    def run(self, maxit=10):
        count = 0
        print(count)
        while count < maxit: 
            try:
                count += 1
                dt = self.step_length()
                self.step(dt)
            except StopIteration:
                break

    def set_init_mesh(self): 

        fd, fh, bbox, pfix, args = self.domain.params
        h, dptol, ttol, Fscale = self.params

        xh = bbox[1] - bbox[0]
        yh = bbox[3] - bbox[2]
        zh = bbox[5] - bbox[4]
        M = int(xh/h)+1
        N = int(yh/h)+1
        Q = int(zh/h)+1


        mg = np.mgrid[
                bbox[0]:bbox[1]:complex(0, M),
                bbox[2]:bbox[3]:complex(0, N),
                bbox[4]:bbox[5]:complex(0, Q)]
        p = np.zeros((M*N*Q, 3), dtype=np.float64)
        p[:, 0] = mg[0].flatten()
        p[:, 1] = mg[1].flatten()
        p[:, 2] = mg[2].flatten()
        p = p[fd(p, *args) < self.geps, :]
        r0 = 1/fh(p, *args)**3
        p = p[np.random.random(p.shape[0]) < r0/np.max(r0), :]
        if pfix is not None:
            p = np.concatenate((pfix, p), axis=0)

        t = self.delaunay(p)
        self.mesh = TetrahedronMesh(p, t)

    def step_length(self):
        return self.dt

    def step(self, dt):
        fd, fh, bbox, pfix, args = self.domain.params
        h, dptol, ttol, Fscale = self.params

        dxdt = self.dx_dt(self.time_elapsed)
        self.mesh.node = self.mesh.node + dt*dxdt

        p = self.mesh.node
        d = fd(p, *args)
        idx = d > 0
        depsx = np.array([self.deps, 0, 0])
        depsy = np.array([0, self.deps, 0])
        depsz = np.array([0, 0, self.deps])
        dgradx = (fd(p[idx, :] + depsx, *args) - d[idx])/self.deps
        dgrady = (fd(p[idx, :] + depsy, *args) - d[idx])/self.deps
        dgradz = (fd(p[idx, :] + depsz, *args) - d[idx])/self.deps
        p[idx, 0] = p[idx, 0] - d[idx]*dgradx
        p[idx, 1] = p[idx, 1] - d[idx]*dgrady
        p[idx, 2] = p[idx, 2] - d[idx]*dgradz
        self.maxmove = np.max(np.sqrt(np.sum(dt*dxdt[d < -self.geps,:]**2, axis=1)))
        self.time_elapsed += dt
        if self.maxmove > ttol*h:
            t = self.delaunay(self.mesh.node)
            self.mesh = TetrahedronMesh(self.mesh.node, t)

        if self.maxmove < dptol*h:
            raise StopIteration

    def dx_dt(self, t):

        fd, fh, bbox, pfix, args = self.domain.params
        h, dptol, ttol, Fscale = self.params

        p = self.mesh.node
        N = p.shape[0]
        edge = self.mesh.ds.edge
        vec = p[edge[:, 0], :] - p[edge[:, 1], :]
        L = np.sqrt(np.sum(vec**2, axis=1))
        hedge = fh(p[edge[:, 1],:]+vec/2, *args) 
        L0 = Fscale*hedge*(np.sum(L**3)/np.sum(hedge**3))**(1/3)
        F = L0 - L
        F[L0-L<0] = 0
        FV = (F/L).reshape((-1,1))*vec

        dxdt = np.zeros((N, 3), dtype=np.float64)
        for i in range(3):
            dxdt[:, i] += np.bincount(edge[:, 0], weights=FV[:, i], minlength=N)
            dxdt[:, i] -= np.bincount(edge[:, 1], weights=FV[:, i], minlength=N)

        if pfix is not None:
            dxdt[0:pfix.shape[0],:] = 0
        return dxdt 

    def delaunay(self, p):
        fd, *_, args = self.domain.params
        d = Delaunay(p)
        t = d.simplices
        pc = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :] + p[t[:, 3], :])/3
        return  t[fd(pc, *args) < - self.geps, :]

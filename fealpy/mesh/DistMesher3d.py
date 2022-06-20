import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from .TetrahedronMesh import TetrahedronMesh 

class DistMesher3d():
    def __init__(self,
            domain, 
            hmin,
            ptol = 0.001,
            ttol = 0.1,
            fscale = 1.1,
            dt = 0.1):

        self.domain = domain
        self.hmin = hmin
        self.ptol = ptol
        self.ttol = ttol
        self.fscale = fscale

        eps = np.finfo(float).eps
        self.geps = 0.1*hmin
        self.deps = np.sqrt(eps)*hmin
        self.dt = 0.1

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.count = 0

    def meshing(self, maxit=1000):
        """
        @brief 运行
        """
        dptol = self.dptol
        self.set_init_mesh()
        count = 0
        while count < maxit: 
            dt = self.step_length()
            self.step(dt)
            count += 1
            if self.maxmove < dptol:
                break
        self.mesh.edge_swap()

    def meshing_with_animation(self, plot=None, axes=None, fname='test.mp4', frames=1000,  interval=50, 
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

        dptol = self.dptol 
        bbox = self.domain.bbox
        lines = LineCollection([], linewidths=linewidths, color=edgecolor)

        def init_func():
            self.set_init_mesh()
            node = self.mesh.entity('node')

            tol = np.max(self.mesh.entity_measure('edge'))
            axes.set_xlim(bbox[0:2])
            axes.set_ylim(bbox[2:4])

            edge = self.mesh.entity('edge')
            lines.set_segments(node[edge])
            axes.add_collection(lines)

            return lines

        def func(n):
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
        self.mesh.edge_swap()

    def set_init_mesh(self): 
        """
        @brief 生成初始网格
        """

        fd = self.domain.signed_dist_function
        fh = self.domain.sizing_function 
        hmin = self.hmin
        box = self.box

        xh = box[1] - box[0]
        yh = box[3] - box[2]
        zh = box[5] - box[4]
        nx = int(xh/hmin) + 1
        ny = int(yh/hmin) + 1
        nz = int(zh/hmin) + 1 

        NN = (nx+1)*(ny+1)*(nz+1)
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:complex(0, nx+1), 
                box[2]:box[3]:complex(0, ny+1),
                box[4]:box[5]:complex(0, nz+1)
                ]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        node = node[fd(node) < self.geps, :]
        r0 = fh(node)**3
        val = np.min(r0)/r0
        NN = len(node)
        node = node[np.random.random(NN) < val]

        fnode = self.domain.facet(0) # 区域中的固定点
        if fnode is not None:
            node = np.concatenate((fnode, node), axis=0)

        cell = self.delaunay(node)
        self.mesh = TetrahedronMesh(node, cell)

    def step_length(self):
        return self.dt

    def step(self, dt):
        """
        @brief 
        """

        fd = self.domain.signed_dist_function
        fh = self.domain.sizing_function 
        hmin = self.hmin
        
        dxdt = self.dx_dt(self.time_elapsed)
        self.mesh.node = self.mesh.node + dt*dxdt

        node = self.mesh.entity('node')
        d = fd(node)
        idx = d > 0
        depsx = np.array([self.deps, 0, 0])
        depsy = np.array([0, self.deps, 0])
        depsy = np.array([0, 0, self.deps])

        dgradx = (fd(node[idx, :] + depsx) - d[idx])/self.deps
        dgrady = (fd(node[idx, :] + depsy) - d[idx])/self.deps
        dgradz = (fd(node[idx, :] + depsz) - d[idx])/self.deps

        node[idx, 0] = node[idx, 0] - d[idx]*dgradx
        node[idx, 1] = node[idx, 1] - d[idx]*dgrady
        node[idx, 2] = node[idx, 2] - d[idx]*dgrady
        self.maxmove = np.max(np.sqrt(np.sum(dt*dxdt[d < -self.geps,:]**2, axis=1))/hmin)
        self.time_elapsed += dt

        if self.maxmove > self.ttol:
            cell = self.delaunay(self.mesh.node)
            self.mesh = TriangleMesh(self.mesh.node, cell)

    def dx_dt(self, t):
        """
        @brief 计算移动步长
        """
        fd = self.domain.signed_dist_function
        fh = self.domain.sizing_function 
        fscale = self.fscale

        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        NN = self.mesh.number_of_nodes()

        v = node[edge[:, 0]] - node[edge[:, 1]]
        L = np.sqrt(np.sum(v**2, axis=1))
        he = fh(node[edge[:, 1]] + v/2) 
        L0 = np.power(np.sum(L**3)/np.sum(he**3), 1/3)*fscale*he
        F = np.minimum(L0 - L, 0)
        FV = (F/L)[:, None]*v

        dxdt = np.zeros((NN, 2), dtype=np.float64)
        np.add.at(dxdt[:, 0], edge[:, 0], FV[:, 0])
        np.add.at(dxdt[:, 1], edge[:, 0], FV[:, 1])
        np.add.at(dxdt[:, 2], edge[:, 0], FV[:, 2])
        np.subtract.at(dxdt[:, 0], edge[:, 1], FV[:, 0])
        np.subtract.at(dxdt[:, 1], edge[:, 1], FV[:, 1])
        np.subtract.at(dxdt[:, 2], edge[:, 1], FV[:, 2])

        fnode = self.domain.facet(0)
        if fnode is not None:
            n = len(fnode)
            dxdt[0:n, :] = 0.0
        return dxdt 

    def delaunay(self, node):
        fd = self.domain.signed_dist_function
        d = Delaunay(node)
        cell = np.asarray(d.simplices, dtype=np.int_)
        bc = (node[cell[:, 0]] + node[cell[:, 1]] + node[cell[:, 2]])/3
        return  cell[fd(bc) < -self.geps, :]

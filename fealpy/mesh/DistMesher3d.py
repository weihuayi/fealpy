import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from .TetrahedronMesh import TetrahedronMesh 

class DistMesher3d():
    def __init__(self,
            domain, 
            hmin,
            ptol = 0.001,
            ttol = 0.01,
            fscale = 1.1,
            dt = 0.1):
        """
        @brief 

        @param[in] domain 三维区域
        @param[in] hmin 最小的边长
        @param[in] ptol
        @param[in] ttol
        @param[in] fscale
        @param[in] dt
        """

        self.domain = domain
        self.hmin = hmin
        self.ptol = ptol
        self.ttol = ttol
        self.fscale = fscale

        eps = np.finfo(float).eps
        self.geps = 0.01*hmin
        self.deps = np.sqrt(eps)*hmin
        self.dt = dt 

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.count = 0


    def set_init_mesh(self): 
        """
        @brief 生成初始网格
        """

        fd = self.domain.signed_dist_function
        fh = self.domain.sizing_function 
        box = self.domain.box

        hmin = self.hmin

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

        node = node[fd(node) < -self.geps, :]

        r0 = fh(node)**3
        val = r0/np.max(r0)
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
        node[idx] = self.domain.projection(node[idx])
        self.maxmove = np.max(dt*np.sqrt(np.sum(dxdt[d < -self.geps,:]**2, axis=1))/hmin)
        self.time_elapsed += dt

        if self.maxmove > self.ttol:
            cell = self.delaunay(self.mesh.node)
            self.mesh = TetrahedronMesh(self.mesh.node, cell)

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

        dxdt = np.zeros((NN, 3), dtype=np.float64)
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
        bc = (node[cell[:, 0]] + node[cell[:, 1]] + node[cell[:, 2]] +
                node[cell[:, 2]])/4
        return  cell[fd(bc) < -self.geps, :]

    def meshing(self, maxit=1000):
        """
        @brief 运行
        """
        ptol = self.ptol
        self.set_init_mesh()
        count = 0
        while count < maxit: 
            fname = "mesh-%05d.vtu"%(count)
            print(fname)
            self.mesh.to_vtk(fname=fname)
            dt = self.step_length()
            self.step(dt)
            count += 1
            if self.maxmove < ptol:
                break

    def meshing_with_animation(self, plot=None, axes=None, 
            fname='test.mp4', frames=1000,  interval=50, 
            edgecolor='k', linewidths=1, aspect='equal', showaxis=False):

        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if plot is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = plt.axes(projection='3d')
        else:
            if isinstance(plot, ModuleType):
                fig, axes = plt.subplots(projection='3d')
            else:
                fig = plot
                if axes is None:
                    axes = fig.gca(projection='3d')

        fig.set_facecolor('white')

        try:
            axes.set_aspect(aspect)
        except NotImplementedError:
            pass

        if showaxis == False:
            axes.set_axis_off()
        else:
            axes.set_axis_on()

        ptol = self.ptol 
        box = self.domain.box
        lines = Line3DCollection([], linewidths=linewidths, color=edgecolor)

        def init_func():
            self.set_init_mesh()
            node = self.mesh.entity('node')
            axes.set_xlim(box[0:2])
            axes.set_ylim(box[2:4])

            edge = self.mesh.entity('edge')
            lines.set_segments(node[edge])
            axes.add_collection3d(lines)

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

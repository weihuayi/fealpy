
import numpy as np

class UniformMesh1d():
    """
    @brief 
    """
    def __init__(self, extent, 
            h=1.0, origin=0.0,
            itype=np.int_, ftype=np.float64):
        self.extent = extent
        self.h = h 
        self.origin = origin

        nx = extent[1] - extent[0]

        self.itype = itype
        self.ftype = ftype

    def geo_dimension(self):
        return 1

    @property
    def node(self):
        GD = self.geo_dimension()
        nx = int(1/self.h)
        node = np.linspace(self.origin, self.origin + nx * self.h, nx+1)
        return node

    def entity_barycenter(self, etype):
        GD = self.geo_dimension()
        nx = int(1/self.h)
        if etype in {'cell', 1}:
            box = [self.origin + self.h / 2, self.origin + (nx - 1) * self.h]
            bc = np.linspace(box[0], box[1], nx)
            return bc

        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError('the entity type `{}` is not correct!'.format(etype))

    def function(self, etype='node', dtype=None, ex=0):
        """
        @brief 返回一个定义在节点或者单元上的数组，元素取值为 0
        """
        nx = int(1/self.h)
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 0}:
            uh = np.zeros(nx + 1, dtype=dtype)
        elif etype in {'cell', 1}:
            uh = np.zeros(nx, dtype=dtype)
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity))

        return uh

    def interpolation(self, f, intertype='node'):
        nx = int(1/self.h)
        node = self.node
        if intertype == 'node':
            F = f(node)
        elif intertype == 'cell':
            bc = self.entity_barycenter('cell')
            F = f(bc)
        return F

    def show_animation(self, fig, axes, box, forward, fname='test.mp4',
                       init=None, fargs=None,
                       frames=1000, lw=2, interval=50):

        import matplotlib.animation as animation

        line, = axes.plot([], [], lw=lw)
        axes.set_xlim(box[0], box[1])
        axes.set_ylim(box[2], box[3])
        x = self.node

        def init_func():
            if callable(init):
                init()
            return line

        def func(n, *fargs):
            uh, t = forward(n)
            line.set_data((x, uh))
            s = "frame=%05d, time=%0.8f" % (n, t)
            print(s)
            axes.set_title(s)
            return line

        ani = animation.FuncAnimation(fig, func, frames=frames,
                                      init_func=init_func,
                                      interval=interval)
        ani.save(fname)
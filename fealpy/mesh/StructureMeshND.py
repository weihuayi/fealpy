import numpy as np
from numpy.linalg import inv

class StructureMeshND:
    def __init__(self, box, N):
        self.box = box
        self.N = N
        self.GD = box.shape[0] 

        self.ftype = np.float
        self.itype = np.int32

    @property
    def node(self):
        N = self.N
        GD = self.GD
        box = self.box
        index = np.ogrid[GD*(slice(0, 1, 1/N), )]
        node = []
        for i in range(GD):
            node.append(sum(map(lambda x: x[0]*x[1], zip(box[i], index))))
        return node

    def interpolation(self, u):
        node = self.node
        return u(node)

    def reciprocal_lattice(self, project_matrix=None, sparse=True):
        """
        返回倒易空间的网格
        """
        N = self.N
        GD = self.GD
        box = self.box

        f = np.fft.fftfreq(N)*N
        f = np.meshgrid(*(GD*(f,)), sparse=sparse)
        rBox = 2*np.pi*inv(box).T
        xi = []
        for i in range(GD):
            xi.append(sum(map(lambda x: x[0]*x[1], zip(rBox[i], f))))
        return xi

    def linear_equation_fft_solver(self, f, 
            cfun=lambda x: 1 + sum(map(lambda y: y**2, x))):
        N = self.N
        GD = self.GD
        xi = self.reciprocal_lattice()
        F = self.interpolation(f) 
        F = np.fft.fftn(F)
        U = F/cfun(xi)
        U = np.fft.ifftn(U).real
        return U

    def error(self, u, U):
        N = self.N
        GD = self.GD
        U0 = self.interpolation(u)
        error = np.sqrt(np.sum((U0 - U)**2)/N**GD)
        return error 




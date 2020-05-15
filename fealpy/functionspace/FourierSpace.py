import numpy as np
from numpy.linalg import inv

class FourierSpace:
    def __init__(self, box, N):
        self.box = box
        self.N = N
        self.GD = box.shape[0] 

        self.ftype = np.float
        self.itype = np.int32

    def number_of_dofs(self):
        return self.N**self.GD


    def interpolation_points(self):
        N = self.N
        GD = self.GD
        box = self.box
        index = np.ogrid[GD*(slice(0, 1, 1/N), )]
        points = []
        for i in range(GD):
            points.append(sum(map(lambda x: x[0]*x[1], zip(box[i], index))))
        return points

    def fourier_interpolation(self, data):
        idx = np.asarray(data[:, :self.GD], dtype=np.int)
        idx[idx<0] += self.N
        F = self.function()
        F[tuple(idx.T)] = data[:, self.GD]
        return np.fft.ifftn(F).real

    def function_norm(self, u):
        dof = self.number_of_dofs()
        val = np.sqrt(np.sum(np.fft.fftn(u))**2/dof).real
        return val

    def interpolation(self, u):
        p = self.interpolation_points()
        return u(p)

    def reciprocal_lattice(self, project_matrix=None, sparse=True,
            return_square=False):
        """
        倒易空间的网格
        """
        N = self.N
        GD = self.GD
        box = self.box

        f = np.fft.fftfreq(N)*N
        f = np.meshgrid(*(GD*(f,)), sparse=sparse)
        rBox = 2*np.pi*inv(box).T
        n = GD
        if project_matrix is not None:
            n = project_matrix.shape[0]
            rBox = project_matrix@rBox
        k = []
        for i in range(n):
            k.append(sum(map(lambda x: x[0]*x[1], zip(rBox[i], f))))

        if return_square:
            return k, sum(map(lambda x: x**2, k))
        else:
            return k

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

    def function(self, dim=None):
        N = self.N
        GD = self.GD
        box = self.box
        shape = GD*(N, )
        if dim is not None:
            shape = (dim, ) + shape
        f = np.zeros(shape, dtype=self.ftype)
        return f

    def error(self, u, U):
        N = self.N
        GD = self.GD
        U0 = self.interpolation(u)
        error = np.sqrt(np.sum((U0 - U)**2)/N**GD)
        return error 




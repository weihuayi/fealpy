import numpy as np
import scipy.fft as spfft 
from numpy.linalg import inv

class FourierSpace:
    def __init__(self, box, N, dft=None):
        self.box = box
        self.N = N
        self.GD = box.shape[0] 

        self.ftype = np.float
        self.itype = np.int32

        if dft is None:
            self.fftn = spfft.fftn
            self.ifftn = spfft.ifftn
            self.fftfreq = spfft.fftfreq  
        else:
            self.fftn = dft.fftn
            self.ifftn = dft.ifftn
            self.fftfreq = dft.fftfreq  

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
        """

        Parameters
        ----------
        data: data[0], data[1]
        """
        idx = data[0]
        idx[idx<0] += self.N
        F = self.function(dtype=data[1].dtype)
        F[tuple(idx.T)] = data[1]
        return self.fftn(F).real

    def function_norm(self, u):
        val = np.sqrt(np.sum(self.ifftn(u))**2).real
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

        f = self.fftfreq(N)*N
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
        F = self.ifftn(F)
        U = F/cfun(xi)
        U = self.fftn(U).real
        return U

    def function(self, dim=None, dtype=None):
        dtype = self.ftype if dtype is None else dtype
        N = self.N
        GD = self.GD
        box = self.box
        shape = GD*(N, )
        if dim is not None:
            shape = (dim, ) + shape
        f = np.zeros(shape, dtype=dtype)
        return f

    def error(self, u, U):
        N = self.N
        GD = self.GD
        U0 = self.interpolation(u)
        error = np.sqrt(np.sum((U0 - U)**2)/N**GD)
        return error 




import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import Sequence, Callable

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator, ScalarConvectionIntegrator, DirichletBC
from fealpy.fem import BilinearForm, LinearForm
from fealpy.pde.diffusion_convection_reaction import PMLPDEModel2d


class NearFieldDataFEMGenerator2d:
    def __init__(self, 
                domain:Sequence[float],
                nx:int,
                ny:int,
                p:int,
                u_inc:str,
                levelset:Callable[[NDArray], NDArray],
                d:Sequence[float],
                k:Sequence[float],
                reciever_points:NDArray):

        self.domain = domain
        self.nx = nx
        self.ny = ny
        self.p = p
        self.u_inc = u_inc
        self.levelset = levelset
        self.d = d 
        self.k = k
        self.reciever_points = reciever_points
        self.mesh = TriangleMesh.interfacemesh_generator(box=self.domain, nx=self.nx, ny=self.ny, phi=self.levelset)
        qf = self.mesh.integrator(p+3, 'cell')
        _, ws = qf.get_quadrature_points_and_weights()
        self.qs = len(ws)

    def get_nearfield_data(self, k, d):

        k_index = (self.k).index(k)
        d_index = (self.d).index(d)
        p =self.p
        pde = PMLPDEModel2d(levelset=self.levelset, 
                                 domain=self.domain, 
                                 qs=self.qs, 
                                 u_inc=self.u_inc,
                                 A=1,
                                 k=self.k[k_index],
                                 d=self.d[d_index],
                                 refractive_index=[1, 1+1/self.k[k_index]**2],
                                 absortion_constant=1.79,
                                 lx=1.0,
                                 ly=1.0
                                 )

        space = LagrangeFESpace(self.mesh, p=p)
        space.ftype = complex

        D = ScalarDiffusionIntegrator(c=pde.diffusion_coefficient, q=p+3)
        C = ScalarConvectionIntegrator(c=pde.convection_coefficient, q=p+3)
        M = ScalarMassIntegrator(c=pde.reaction_coefficient, q=p+3)
        f = ScalarSourceIntegrator(pde.source, q=p+3)

        b = BilinearForm(space)
        b.add_domain_integrator([D, C, M])

        l = LinearForm(space)
        l.add_domain_integrator(f)

        A = b.assembly()
        F = l.assembly()
        bc = DirichletBC(space, pde.dirichlet) 
        uh = space.function(dtype=np.complex128)
        A, F = bc.apply(A, F, uh)
        uh[:] = spsolve(A, F)
        return uh

    def data_for_dsm(self, k, d):
        
        uh = self.get_nearfield_data(k=k, d=d)
        b = self.mesh.point_to_bc(self.reciever_points)
        location = self.mesh.location(self.reciever_points)
        data = np.zeros(len(b), dtype=np.complex128)
        for i in range (len(b)):
            data[i] = uh(b[i])[location[i]]
        return data
    
    def save(self, scatterer_index:int):

        k_values = self.k
        d_values = self.d
        data_dict = {}
        for i in range (len(k_values)):
            for j in range(len(d_values)):
                k_name = f'k={k_values[i]}'
                d_name = d_values[j]
                name = f"{k_name}, d={d_name}"
                data_dict[name] = self.data_for_dsm(k=k_values[i], d=d_values[j])
        filename = f"data_for_dsm_{scatterer_index}.npz"
        np.savez(filename, **data_dict)

    def get_specified_data(self, k, d):

        k_index = (self.k).index(k)
        d_index = (self.d).index(d)
        file_index = k_index * len(self.d) + d_index
        loaded_data = np.load('data_for_dsm.npz', allow_pickle=True)
        keys = loaded_data.files
        val = loaded_data[keys[file_index]]
        return val

    def visualization_of_nearfield_data(self, k, d):

        uh = self.get_nearfield_data(k=k, d=d)
        fig = plt.figure()
        bc = np.array([[1/3, 1/3, 1/3]], dtype=np.float64)
        value = uh(bc)
        self.mesh.add_plot(plt, cellcolor=value[0, ...].real, linewidths=0)
        self.mesh.add_plot(plt, cellcolor=value[0, ...].imag, linewidths=0)
        
        axes = fig.add_subplot(1, 3, 1)
        self.mesh.add_plot(axes)
        axes = fig.add_subplot(1, 3, 2, projection='3d')
        self.mesh.show_function(axes, np.real(uh))
        axes = fig.add_subplot(1, 3, 3, projection='3d')
        self.mesh.show_function(axes, np.imag(uh))
        plt.show()
        
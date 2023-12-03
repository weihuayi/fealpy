import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import Sequence, Callable

from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator, ScalarConvectionIntegrator, DirichletBC
from fealpy.fem import BilinearForm, LinearForm
from fealpy.pde.diffusion_convection_reaction import PMLPDEModel2d


def quadranglemesh_point_location_and_bc(p, domain, nx, ny):

    x = p[..., 0]
    y = p[..., 1]
    cell_length_x = (domain[1] - domain[0])/nx
    cell_length_y = (domain[3] - domain[2])/ny
    index_x = (x - domain[0])//cell_length_x
    index_y = (y - domain[2])//cell_length_y
    
    location = int(index_x * ny + index_y)
    cell_x = domain[0] + (location // ny) * cell_length_x
    cell_y = domain[2] + (location % nx) * cell_length_y   
    
    bc_x = np.array([[(x - cell_x)/cell_length_x, (cell_x - x)/cell_length_x + 1 ]], dtype=np.float64)
    bc_y = np.array([[(y - cell_y)/cell_length_y, (cell_y - y)/cell_length_y + 1]], dtype=np.float64)
    bc = (bc_x, bc_y)
    return location, bc

class NearFieldDataFEMGenerator2d:
    def __init__(self, 
                domain:Sequence[float],
                mesh:str,
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
        if mesh not in ['InterfaceMesh', 'QuadrangleMesh']:
            raise ValueError("Invalid value for 'mesh'. Choose from 'InterfaceMesh' or 'QuadrangleMesh'.")
        else:
            if mesh =='InterfaceMesh':
                self.mesh = TriangleMesh.interfacemesh_generator(box=self.domain, nx=self.nx, ny=self.ny, phi=self.levelset)  
                self.meshtype = 'InterfaceMesh'
            else:
                self.mesh = QuadrangleMesh.from_box(box=self.domain, nx=self.nx, ny=self.ny)
                self.meshtype = 'QuadrangleMesh'
        self.mesh.ftype = complex
        self.d = d 
        self.k = k
        self.reciever_points = reciever_points
        qf = self.mesh.integrator(p+2, 'cell')
        self.bc, ws = qf.get_quadrature_points_and_weights()
        self.qs = len(ws)

    def get_nearfield_data(self, k, d):

        k_index = (self.k).index(k)
        d_index = (self.d).index(d)
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

        space = LagrangeFESpace(self.mesh, p=self.p)

        D = ScalarDiffusionIntegrator(c=pde.diffusion_coefficient, q=self.p+2)
        C = ScalarConvectionIntegrator(c=pde.convection_coefficient, q=self.p+2)
        M = ScalarMassIntegrator(c=pde.reaction_coefficient, q=self.p+2)
        f = ScalarSourceIntegrator(pde.source, q=self.p+2)

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

        reciever_points = self.reciever_points
        data_length = reciever_points.shape[0]
        uh = self.get_nearfield_data(k=k, d=d)
        if self.meshtype =='InterfaceMesh':
            b = self.mesh.point_to_bc(reciever_points)
            location = self.mesh.location(reciever_points)
            data = np.zeros(len(data_length), dtype=np.complex128)
            for i in range (len(data_length)):
                data[i] = uh(b[i])[location[i]]
        else:
            data = np.zeros(data_length, dtype=np.complex128)
            for i in range(data_length):
                location, b = quadranglemesh_point_location_and_bc(reciever_points[i], self.domain, self.nx, self.ny)
                u = uh(b).reshape(-1)
                data[i] = u[location]
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
        value = uh(self.bc)
        self.mesh.add_plot(plt, cellcolor=value[0, ...].real, linewidths=0)
        self.mesh.add_plot(plt, cellcolor=value[0, ...].imag, linewidths=0)
        
        axes = fig.add_subplot(1, 3, 1)
        self.mesh.add_plot(axes)
        axes = fig.add_subplot(1, 3, 2, projection='3d')
        self.mesh.show_function(axes, np.real(uh))
        axes = fig.add_subplot(1, 3, 3, projection='3d')
        self.mesh.show_function(axes, np.imag(uh))
        plt.show()
        
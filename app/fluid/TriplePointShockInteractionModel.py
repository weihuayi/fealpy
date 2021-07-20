
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class TriplePointShockInteractionModel:

    def __init__(self):
        self.domain = [0, 7, 0, 3]

    def space_mesh(self, p, NS=0):
        from fealpy.mesh import MeshFactory as MF
        mesh = MF.boxmesh2d(self.domain, nx=70, ny=30, p=p, meshtype='quad') 
        mesh.uniform_refine(NS)
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NCN = mesh.number_of_corner_nodes()
        cell = mesh.entity('cell')
        valence = np.zeros(NN, dtype=np.int) 
        np.add.at(valence, cell, 1)
        dof = np.zeros(NN, dtype=np.int)
        idx = np.arange(NN)
        flag = (valence == 2) & (idx < NCN)
        dof[flag] = 1
        flag = (valence == 1) & (idx > NCN) & (idx < NCN + (p-1)*NE)
        dof[flag] = 1
        flag = (valence == 1) & (idx > NCN + (p-1)*NE)
        dof[flag] = 2
        flag = (valence > 2) & (idx < NCN)
        dof[flag] = 2
        flag = (valence == 2) & (idx > NCN) & (idx < NCN + (p-1)*NE)
        dof[flag] = 2
        mesh.nodedata['dof'] = dof

        flag = dof == 1
        n = flag.sum()
        en = np.zeros((n, 2), dtype=np.float64)
        node = mesh.entity('node')[flag]
        flag = np.abs(node[:, 0] - 0.0) < 1e-12
        en[flag, 0] = -1.0
        flag = np.abs(node[:, 0] - 7.0) < 1e-12
        en[flag, 0] = 1.0

        flag = np.abs(node[:, 1] - 0.0) < 1e-12
        en[flag, 1] = -1.0
        flag = np.abs(node[:, 1] - 3.0) < 1e-12
        en[flag, 1] = 1.0
        mesh.meshdata['bd_normal'] = en
        return mesh

    def time_mesh(self, NT=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, 1, NT)
        return timeline

    def subdomain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.zeros(p.shape[:-1], dtype=p.dtype)
        flag[x < 1] = 1  
        flag[(x > 1) & (y < 1.5)] = 2
        flag[(x > 1) & (y > 1.5)] = 3 
        return flag

    def init_rho(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.0
        val[(x > 1) & (y < 1.5)] = 1.0
        val[(x > 1) & (y > 1.5)] = 0.125
        return val

    def init_velocity(self, v):
        v[:] = 0.0

    def init_energe(self, e):
        p = e.space.interpolation_points()
        x = p[..., 0]
        y = p[..., 1]
        e[x < 1] = 1.0/(1.5 - 1)/1.0
        e[(x > 1) & (y < 1.5)] = 0.1/(1.4-1)/1.0
        e[(x > 1) & (y > 1.5)] = 0.1/(1.5-1)/0.125 

    def adiabatic_index(self, p):
        """
        Notes
        -----
        绝热指数
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.5
        val[(x > 1) & (y < 1.5)] = 1.4 
        val[(x > 1) & (y > 1.5)] = 1.5 
        return val

    def stress(self, bc, e, rho, gamma, mesh0, mesh1):
        """

        Notes

        rho: (NC, ) 每个单元上的初始密度
        gamma: (NC, ) 每个单元上的绝热指数
        """

        J0 = mesh0.jacobi_matrix(bc) # (NQ, NC, GD, GD)
        J1 = mesh1.jacobi_matrix(bc) # (NQ, NC, GD, GD)
        J0 = np.linalg.det(J0) # (NQ, NC)
        J1 = np.linalg.det(J1) # (NQ, NC)

        val = J0
        val /= J1 
        val *= e.value(bc)
        val *= (1 - gamma)*rho
        return val 

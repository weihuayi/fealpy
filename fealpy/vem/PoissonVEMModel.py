import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..vem import doperator
from .integral_alg import PolygonMeshIntegralAlg


class PoissonVEMModel():
    def __init__(self, pde, mesh, p=1, q=4):
        """
        Initialize a Poisson virtual element model. 

        Parameters
        ----------
        self : PoissonVEMModel object
        pde :  PDE Model object
        mesh : PolygonMesh object
        p : int
        
        See Also
        --------

        Notes
        -----
        """
        self.space =VirtualElementSpace2d(mesh, p) 
        self.mesh = self.space.mesh
        self.pde = pde 
        self.uh = self.space.function()
        self.area = self.space.smspace.area 
        self.integrator = mesh.integrator(q)

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.space.smspace.barycenter)

        self.uI = self.space.interpolation(pde.solution, self.integralalg.integral)

        self.mat = doperator.basic_matrix(self.space, self.area)

    def reinit(self, mesh, p):
        self.space =VirtualElementSpace2d(mesh, p) 
        self.mesh = self.space.mesh
        self.uh = self.space.function()
        self.area = self.space.smspace.area 

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.space.smspace.barycenter)

        self.uI = self.space.interpolation(self.pde.solution, self.integralalg.integral)

        self.mat = doperator.basic_matrix(self.space, self.area)


    def project_to_smspace(self, uh=None):
        p = self.space.p
        cell2dof, cell2dofLocation = self.space.dof.cell2dof, self.space.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        if uh is None:
            g = lambda x: x[0]@self.uh[x[1]]
        else:
            g = lambda x: x[0]@uh[x[1]]
        S = self.space.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.mat.PI1, cd))))
        return S

    def recover_estimate(self, uh=None, rtype='simple', residual=True):
        """
        estimate the recover-type error 

        Parameters
        ----------
        self : PoissonVEMModel object
        rtype : str
            'simple':
            'area'
            'inv_area'

        See Also
        --------

        Notes
        ----- 

        """
        space = self.space
        mesh = space.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        barycenter = space.smspace.barycenter 

        h = space.smspace.h 
        area = space.smspace.area
        ldof = space.smspace.number_of_local_dofs()
            
        # project the vem solution into linear polynomial space
        idx = np.repeat(range(NC), NV)
        if uh is None:
            S = self.project_to_smspace(self.uh)
        else:
            S = self.project_to_smspace(uh)
        grad = S.grad_value(barycenter)

        S0 = space.smspace.function() 
        S1 = space.smspace.function()
        p2c = mesh.ds.node_to_cell()
        try: 
            isSubDomain = self.pde.subdomain(barycenter)
            for isFlag in isSubDomain:
                isSubIdx = np.repeat(isFlag, NV)
                M = p2c[:, isFlag]
                sa = area[isFlag]
                if rtype is 'simple':
                    d = p2c.sum(axis=1)
                    ruh = np.asarray((M@grad[isFlag])/d.reshape(-1, 1))
                elif rtype is 'area':
                    d = p2c@area
                    ruh = np.asarray((M@(grad[isFlag]*sa.reshape(-1, 1)))/d.reshape(-1, 1))
                elif rtype is 'inv_area':
                    d = p2c@(1/area)
                    ruh = np.asarray((M@(grad[isFlag]/sa.reshape(-1, 1)))/d.reshape(-1, 1))
                else:
                    raise ValueError("I have note code method: {}!".format(rtype))

                for i in range(3):
                    S0[i::ldof] += np.bincount(idx[isSubIdx], weights=self.B[i, isSubIdx]*ruh[cell[isSubIdx], 0], minlength=NC)
                    S1[i::ldof] += np.bincount(idx[isSubIdx], weights=self.B[i, isSubIdx]*ruh[cell[isSubIdx], 1], minlength=NC)

        except  AttributeError:
            if rtype is 'simple':
                d = p2c.sum(axis=1)
                ruh = np.asarray((p2c@grad)/d.reshape(-1, 1))
            elif rtype is 'area':
                d = p2c@area
                ruh = np.asarray((p2c@(grad*area.reshape(-1, 1)))/d.reshape(-1, 1))
            elif rtype is 'inv_area':
                d = p2c@(1/area)
                ruh = np.asarray((p2c@(grad/area.reshape(-1,1)))/d.reshape(-1, 1))
            else:
                raise ValueError("I have note code method: {}!".format(rtype))

            for i in range(ldof):
                S0[i::ldof] = np.bincount(idx, weights=self.mat.B[i, :]*ruh[cell, 0], minlength=NC)
                S1[i::ldof] = np.bincount(idx, weights=self.mat.B[i, :]*ruh[cell, 1], minlength=NC)

        try:
            k = self.pde.diffusion_coefficient(barycenter)
        except  AttributeError:
            k = np.ones(NC) 

        node = mesh.node
        gx = S0.value(node[cell], idx) - np.repeat(grad[:, 0], NV)
        gy = S1.value(node[cell], idx) - np.repeat(grad[:, 1], NV)
        eta = k*np.bincount(idx, weights=gx**2+gy**2)/NV*area

        if residual:
            fh = self.integralalg.fun_integral(self.pde.source, True)/self.area
            g0 = S0.grad_value(barycenter)
            g1 = S1.grad_value(barycenter)
            eta += k*(fh + k*(g0[:, 0] + g1[:, 1]))**2*area**2

        return np.sqrt(eta)

    def get_left_matrix(self):
        space = self.space
        area = self.area
        try:
            a = self.pde.diffusion_coefficient
            return doperator.stiff_matrix(space, area, cfun=a, mat=self.mat)
        except AttributeError:
            return doperator.stiff_matrix(space, area, mat=self.mat)

    def get_right_vector(self):
        f = self.pde.source
        integral = self.integralalg.integral 
        return doperator.source_vector(
                integral,
                f, 
                self.space,
                self.mat.PI0)

    def solve(self):
        uh = self.uh
        bc = DirichletBC(self.space, self.pde.dirichlet)
        self.A, b = solve(self, uh, dirichlet=bc, solver='direct')

    def l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self):
        u = self.pde.solution
        S = self.project_to_smspace(self.uh)
        uh = S.value
        return self.integralalg.L2_error(u, uh)

    def H1_semi_error(self):
        gu = self.pde.gradient
        S = self.project_to_smspace(self.uh)
        guh = S.grad_value
        return self.integralalg.L2_error(gu, guh)

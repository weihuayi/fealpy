import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..vemmodel import doperator
from .integral_alg import PolygonMeshIntegralAlg


class PoissonVEMModel():
    def __init__(self, pde, mesh, p, integrator):
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
        self.vemspace =VirtualElementSpace2d(mesh, p) 
        self.mesh = self.vemspace.mesh
        self.pde = pde  
        print('pde',self.pde)
        self.uh = self.vemspace.function() 
        self.area = self.vemspace.smspace.area
        self.integrator = integrator

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.vemspace.smspace.barycenter)

        #self.uI = self.vemspace.interpolation(pde.solution, self.integralalg.integral)

        self.mat = doperator.basic_matrix(self.vemspace, self.area)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.vemspace.p
        self.vemspace = VirtualElementSpace2d(mesh, p) 
        self.mesh = self.vemspace.mesh
        self.uh = self.vemspace.function() 
        self.area = self.vemspace.smspace.area

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.vemspace.smspace.barycenter)
        #self.uI = self.vemspace.interpolation(self.pde.solution, self.integralalg.integral)

        self.mat = doperator.basic_matrix(self.vemspace, self.area)

    def project_to_smspace(self, uh=None):
        p = self.vemspace.p
        cell2dof, cell2dofLocation = self.vemspace.dof.cell2dof, self.vemspace.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@self.uh[x[1]]
        S = self.vemspace.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.mat.PI1, cd))))
        return S

    def recover_estimate(self, rtype='simple', residual=True):
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
        uh = self.uh
        vemspace = self.vemspace
        mesh = vemspace.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        barycenter = vemspace.smspace.barycenter 

        h = vemspace.smspace.h 
        area = vemspace.smspace.area
        ldof = vemspace.smspace.number_of_local_dofs()
            
        # project the vem solution into linear polynomial space
        idx = np.repeat(range(NC), NV)
        S = self.project_to_smspace(self.uh)
        grad = S.grad_value(barycenter)

        S0 = vemspace.smspace.function() 
        S1 = vemspace.smspace.function()
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
        vemspace = self.vemspace
        area = self.area
        try:
            f = self.pde.diffusion_coefficient
            return doperator.stiff_matrix(vemspace, area, cfun=f, mat=self.mat)
        except AttributeError:
            return doperator.stiff_matrix(vemspace, area, mat=self.mat)

    def get_right_vector(self):
        f = self.pde.source
        integral = self.integralalg.integral        
        return doperator.source_vector(
                integral,
                f, 
                self.vemspace,
                self.mat.PI0)

    def solve(self):
        uh = self.uh
        #bc = DirichletBC(self.vemspace, self.pde.dirichlet) 
        bc = DirichletBC(self.vemspace, self.is_boundary_dof)     
        self.A, b = solve(self, uh, dirichlet=bc, solver='direct')

    def is_boundary_dof(self, p):
        isBdDof = np.zeros(p.shape[0], dtype=np.bool_)
        isBdDof[0] = True
        return isBdDof
    
    

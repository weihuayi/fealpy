import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..vemmodel import doperator 
from ..functionspace import FunctionNorm


class PoissonVEMModel():
    def __init__(self, model, mesh, p=1, integrator=None, quadtree=None):
        """
        Initialize a Poisson virtual element model. 

        Parameters
        ----------
        self : PoissonVEMModel object
        model :  PDE Model object
        mesh : PolygonMesh object
        p : int
        
        See Also
        --------

        Notes
        -----
        """
        self.V =VirtualElementSpace2d(mesh, p) 
        self.mesh = self.V.mesh
        self.quadtree = quadtree
        self.model = model  
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(model.solution)
        self.area = self.V.smspace.area 

        self.error = FunctionNorm(integrator, self.area)

        self.H = doperator.matrix_H(self.V)

        self.D = doperator.matrix_D(self.V, self.H)
        self.B = doperator.matrix_B(self.V)
        self.C = doperator.matrix_C(self.V, self.B, self.D, self.H, self.area)
        self.G = doperator.matrix_G(self.V, self.B, self.D)

        self.PI0 = doperator.matrix_PI_0(self.V, self.H, self.C)
        self.PI1 = doperator.matrix_PI_1(self.V, self.G, self.B)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = VirtualElementSpace2d(mesh, p) 
        self.mesh = self.V.mesh
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.smspace.area
        self.error.area = self.area 

        self.H = doperator.matrix_H(self.V)
        self.D = doperator.matrix_D(self.V, self.H)
        self.B = doperator.matrix_B(self.V)
        self.C = doperator.matrix_C(self.V, self.B, self.D, self.H, self.area)

        self.PI0 = doperator.matrix_PI_0(self.V, self.H, self.C)
        self.PI1 = doperator.matrix_PI_1(self.V, self.G, self.B)

    def project_to_smspace(self, uh=None):
        p = self.V.p
#        if p == 1:
#            V = self.V
#            mesh = V.mesh
#            NC = mesh.number_of_cells()
#            NV = mesh.number_of_vertices_of_cells()
#            cell = mesh.ds.cell
#            ldof = V.smspace.number_of_local_dofs()
#
#            S = self.V.smspace.function()
#            if uh is None:
#                uh = self.uh
#            idx = np.repeat(range(NC), NV)
#            for i in range(3):
#                S[i::ldof] = np.bincount(idx, weights=self.B[i, :]*uh[cell], minlength=NC)
#        else:
        cell2dof, cell2dofLocation = self.V.dof.cell2dof, self.V.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@self.uh[x[1]]
        S = self.V.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def recover_estimate(self, rtype='simple'):
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
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        barycenter = V.smspace.barycenter 

        h = V.smspace.h 
        area = V.smspace.area
        ldof = V.smspace.number_of_local_dofs()
            
        # project the vem solution into linear polynomial space
        idx = np.repeat(range(NC), NV)
        S = self.S 
        grad = S.grad_value(barycenter)

        S0 = V.smspace.function() 
        S1 = V.smspace.function()
        p2c = mesh.ds.point_to_cell()
        try: 
            isSubDomain = self.model.subdomain(barycenter)
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
                S0[i::ldof] = np.bincount(idx, weights=self.B[i, :]*ruh[cell, 0], minlength=NC)
                S1[i::ldof] = np.bincount(idx, weights=self.B[i, :]*ruh[cell, 1], minlength=NC)

        try:
            k = self.model.diffusion_coefficient(barycenter)
        except  AttributeError:
            k = np.ones(NC) 

        point = mesh.point
        gx = S0.value(point[cell], idx) - np.repeat(grad[:, 0], NV)
        gy = S1.value(point[cell], idx) - np.repeat(grad[:, 1], NV)
        eta = np.sqrt(k*np.bincount(idx, weights=gx**2+gy**2)/NV*area)
        return eta

    def get_left_matrix(self):
        V = self.V
        area = self.area
        try:
            f = self.model.diffusion_coefficient
            return doperator.stiff_matrix(V, area, cfun=f, vem=self)
        except AttributeError:
            return doperator.stiff_matrix(V, area, vem=self)

    def get_right_vector(self):
        V = self.V
        area = self.area
        f = self.model.source
        return doperator.source_vector(f, V, area, vem=self)

    def solve(self):
        uh = self.uh
        bc = DirichletBC(self.V, self.model.dirichlet)
        self.A, b = solve(self, uh, dirichlet=bc, solver='direct')
        self.S = self.project_to_smspace(uh)
    
    def l2_error(self):
        u = self.model.solution
        uh = self.uh
        return self.error.l2_error(u, uh)

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self):
        u = self.model.solution
        uh = self.S.value
        return self.error.L2_error(u, uh, self.quadtree, barycenter=False)

    def H1_semi_error(self):
        gu = self.model.gradient
        guh = self.S.grad_value
        return self.error.L2_error(gu, guh, self.quadtree, barycenter=False)


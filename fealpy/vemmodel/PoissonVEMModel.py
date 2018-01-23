import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..quadrature import QuadrangleQuadrature 
from ..vemmodel import form 

from timeit import default_timer as timer

class PoissonVEMModel():
    def __init__(self, model, mesh, p=1):
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
        self.model = model  
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(model.solution)
        self.area = self.V.smspace.area 

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = VirtualElementSpace2d(mesh, p) 
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.smspace.area

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

        try:
            k = self.model.diffusion_coefficient(barycenter)
        except  AttributeError:
            k = np.ones(NC) 
            
        # project the vem solution into linear polynomial space
        S = V.project_to_smspace(self.uh, self.B)

        p2c = mesh.ds.point_to_cell()
        try: 
            isSubDomain = self.model.subdomain(barycenter)
            eta = np.zeros((NC, ), dtype=np.float)
            for isFlag in isSubDomain:
                isSubIdx = np.repeat(isFlag, NV)
                M = p2c[:, isFlag]
                sa = area[isFlag]
                if rtype is 'simple':
                    d = p2c.sum(axis=1)
                    ruh = np.asarray((M@S[isFlag, 1:3])/d.reshape(-1, 1))
                elif rtype is 'area':
                    d = p2c@area
                    ruh = np.asarray((M@(S[isFlag, 1:3]*sa.reshape(-1, 1)))/d.reshape(-1, 1))
                elif rtype is 'inv_area':
                    d = p2c@(1/area)
                    ruh = np.asarray((M@(S[isFlag, 1:3]/sa.reshape(-1, 1)))/d.reshape(-1, 1))
                else:
                    raise ValueError("I have note code method: {}!".format(rtype))

                idx = np.repeat(range(NC), NV)
                S1 = np.zeros((NC, 3), dtype=np.float)
                S2 = np.zeros((NC, 3), dtype=np.float)
                for i in range(3):
                    S1[:, i] = np.bincount(idx[isSubIdx], weights=V.B[i, isSubIdx]*ruh[cell[isSubIdx], 0], minlength=NC)
                    S2[:, i] = np.bincount(idx[isSubIdx], weights=V.B[i, isSubIdx]*ruh[cell[isSubIdx], 1], minlength=NC)

                point = mesh.point
                phi1 = (point[cell[isSubIdx], 0] - np.repeat(barycenter[isFlag, 0], NV[isFlag]))/np.repeat(h[isFlag], NV[isFlag])
                phi2 = (point[cell[isSubIdx], 1] - np.repeat(barycenter[isFlag, 1], NV[isFlag]))/np.repeat(h[isFlag], NV[isFlag])

                gx = np.repeat(S1[isFlag, 0], NV[isFlag])+ np.repeat(S1[isFlag, 1], NV[isFlag])*phi1 + \
                    np.repeat(S1[isFlag, 2], NV[isFlag])*phi2 - np.repeat(S[isFlag, 1], NV[isFlag])
                gy = np.repeat(S2[isFlag,  0], NV[isFlag])+ np.repeat(S2[isFlag, 1], NV[isFlag])*phi1 + \
                    np.repeat(S2[isFlag, 2], NV[isFlag])*phi2 - np.repeat(S[isFlag, 2], NV[isFlag])
                eta += np.sqrt(k*np.bincount(idx[isSubIdx], weights=gx**2+gy**2, minlength=NC)/NV*area)
        except  AttributeError:
            if rtype is 'simple':
                d = p2c.sum(axis=1)
                ruh = np.asarray((p2c@S[:, 1:3])/d.reshape(-1, 1))
            elif rtype is 'area':
                d = p2c@area
                ruh = np.asarray((p2c@(S[:, 1:3]*area.reshape(-1, 1)))/d.reshape(-1, 1))
            elif rtype is 'inv_area':
                d = p2c@(1/area)
                ruh = np.asarray((p2c@(S[:, 1:3]/area.reshape(-1,1)))/d.reshape(-1, 1))
            else:
                raise ValueError("I have note code method: {}!".format(rtype))

            S1 = V.project_to_smspace(ruh[:, 0])
            S2 = V.project_to_smspace(ruh[:, 1]) 

            point = mesh.point
            cellIdx = np.repeat(range(NC), NV)
            gx = S1.value(point[cell], cellIdx) - np.repeat(S[:, 1], NV)
            gy = S2.value(point[cell], cellIdx) - np.repeat(S[:, 2], NV)
            eta = np.sqrt(k*np.bincount(np.repeat(range(NC), NV), weights=gx**2+gy**2)/NV*area)
        return eta

    def get_left_matrix(self):
        return form.stiff_matrix(self)

    def get_right_vector(self):
        return form.source_vector(self)

    def solve(self):
        uh = self.uh
        bc = DirichletBC(self.V, self.model.dirichlet)
        self.A, b = solve(self, uh, dirichlet=bc, solver='direct')

    def l2_error(self):
        uh = self.uh
        uI = self.uI 
        return np.sqrt(np.sum((uh - uI)**2)/len(uI))

    def interpolation_error(self):
        A = self.A
        uh = self.uh
        uI = self.uI 
        e = uh - uI
        return np.sqrt(e@A@e)

    def L2_error(self, order, quadtree):
        V = self.V
        mesh = V.mesh
        model = self.model

        NV = mesh.number_of_vertices_of_cells()
        NC = mesh.number_of_cells()
        cell2dof,cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation

        #Project vem function to polynomial function 
        sh = V.project_to_smspace(self.uh)

        qf = QuadrangleQuadrature(2)
        nQuad = qf.get_number_of_quad_points()
        e = np.zeros((NC,), dtype=np.float)

        cell = quadtree.leaf_cell()
        point = quadtree.point
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            p = quadtree.bc_to_point(bc)
            uhval = sh.value(p)
            uval = self.model.solution(p)
            e += w*(uhval - uval)*(uhval - uval)
        e *= self.area
        return np.sqrt(e.sum()) 


#        e = (self.uh - self.uI)**2
#        area = V.smspace.area
#        e = e[cell]/np.repeat(NV, NV)
#        e = np.bincount(np.repeat(range(NC), NV), weights=e, minlength=NC)*area
#        return np.sqrt(np.sum(e)) 

    def H1_error(self):
        pass

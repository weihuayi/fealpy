import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..quadrature import QuadrangleQuadrature 

from timeit import default_timer as timer

class PoissonVEMModel():
    def __init__(self, model, mesh, p=1, dtype=np.float):

        self.V =VirtualElementSpace2d(mesh, p, dtype) 
        self.model = model  
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(model.solution)
        self.area = self.V.smspace.area 


        self.dtype=dtype

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = VirtualElementSpace2d(mesh, p, self.dtype) 
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.smspace.area

    def recover_estimate(self, rtype='simple'):
        uh = self.uh
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        barycenter = V.smspace.barycenter 

        B = self.B 
        h = V.smspace.h 
        area = V.smspace.area

        try:
            k = self.model.diffusion_coefficient(barycenter)
        except  AttributeError:
            k = np.ones(NC) 
            
        # project the vem solution into linear polynomial space
        S = np.zeros((NC, 3), dtype=self.dtype)
        idx = np.repeat(range(NC), NV)
        for i in range(3):
            S[:, i] = np.bincount(idx, weights=B[i, :]*uh[cell], minlength=NC)
        S /=h.reshape(-1, 1)

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

                S1 = np.zeros((NC, 3), dtype=self.dtype)
                S2 = np.zeros((NC, 3), dtype=self.dtype)
                for i in range(3):
                    S1[:, i] = np.bincount(idx[isSubIdx], weights=B[i, isSubIdx]*ruh[cell[isSubIdx], 0], minlength=NC)
                    S2[:, i] = np.bincount(idx[isSubIdx], weights=B[i, isSubIdx]*ruh[cell[isSubIdx], 1], minlength=NC)

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

            S1 = np.zeros((NC, 3), dtype=self.dtype)
            S2 = np.zeros((NC, 3), dtype=self.dtype)
            for i in range(3):
                S1[:, i] = np.bincount(idx, weights=B[i, :]*ruh[cell, 0], minlength=NC)
                S2[:, i] = np.bincount(idx, weights=B[i, :]*ruh[cell, 1], minlength=NC)

            point = mesh.point
            phi1 = (point[cell, 0] - np.repeat(barycenter[:, 0], NV))/np.repeat(h, NV)
            phi2 = (point[cell, 1] - np.repeat(barycenter[:, 1], NV))/np.repeat(h, NV)

            gx = np.repeat(S1[:, 0], NV)+ np.repeat(S1[:, 1], NV)*phi1 + \
                np.repeat(S1[:, 2], NV)*phi2 - np.repeat(S[:, 1], NV)
            gy = np.repeat(S2[:,  0], NV)+ np.repeat(S2[:, 1], NV)*phi1 + \
                np.repeat(S2[:, 2], NV)*phi2 - np.repeat(S[:, 2], NV)
            eta = np.sqrt(k*np.bincount(np.repeat(range(NC), NV), weights=gx**2+gy**2)/NV*area)
        return eta

    def get_left_matrix(self):
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()

        p = V.p
        area = V.smspace.area
        h = V.smspace.h

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        smldof = V.smspace.number_of_local_dofs()

        A = coo_matrix((gdof, gdof), dtype=self.dtype)

        cell2dof, cell2dofLocation = V.cell_to_dof()
        cell = mesh.ds.cell
        point = mesh.point

        if p == 1:
            NV = mesh.number_of_vertices_of_cells()
            B = np.zeros((smldof, cell2dof.shape[0]), dtype=self.dtype) 
            B[0, :] = 1/np.repeat(NV, NV)
            B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1,-1)
            self.B = B
            bc = np.repeat(V.smspace.barycenter, NV, axis=0)
            D = np.ones((cell2dof.shape[0], smldof), dtype=self.dtype)
            D[:, 1:] = (point[cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            G = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])

            BB = np.hsplit(B, cell2dofLocation[1:-1])
            DD = np.vsplit(D, cell2dofLocation[1:-1])
            cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

            
            f1 = lambda x: (x[1].T@G@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
            f2 = lambda x: np.repeat(x, x.shape[0]) 
            f3 = lambda x: np.tile(x, x.shape[0])
            f4 = lambda x: x.flatten()

            try:
                barycenter = V.smspace.barycenter 
                k = self.model.diffusion_coefficient(barycenter)
            except  AttributeError:
                k = np.ones(NC) 

            K = list(map(f1, zip(DD, BB, k)))
            I = np.concatenate(list(map(f2, cd)))
            J = np.concatenate(list(map(f3, cd)))
            val = np.concatenate(list(map(f4, K)))
            A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=self.dtype)
            end = timer()
        
        else:
            raise ValueError("I have not code the vem with degree > 1!")

        return A

    def get_right_vector(self):

        V = self.V
        mesh = V.mesh
        model = self.model

        ldof = V.number_of_local_dofs()
        bb = np.zeros(ldof.sum(), dtype=self.dtype)
        point = mesh.point
        NV = mesh.number_of_vertices_of_cells()
        F = model.source(point)
        area = V.smspace.area
        cell2dof, cell2dofLocation = V.cell_to_dof()
        bb = F[cell2dof]/np.repeat(NV, NV)*np.repeat(area, NV)
        gdof = V.number_of_global_dofs()
        b = np.bincount(cell2dof, weights=bb, minlength=gdof)
        return b

    def get_neuman_vector(self):
        pass

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
        cell2dof = V.cell_to_dof()

        #Project vem function to polynomial function 
        uh = self.uh
        sh = self.V.smspace.function()
        idx = np.repeat(range(NC), NV)
        for i in range(3):
            sh[:, i] = np.bincount(idx, weights=B[i, :]*uh[cell2dof], minlength=NC)

        qf = QuadrangleQuadrature(2)
        nQuad = qf.get_number_of_quad_points()
        e = np.zeros((NC,), dtype=self.dtype)

        cell = quadtree.leaf_cell()
        point = quadtree.point
        for i in range(nQuad):
            lambda_k, w_k = qf.get_gauss_point_and_weight(i)
            p = mesh.bc_to_point(lambda_k)
            uhval = sh.value(p)
            uval = self.model.solution(p)
            e += w_k*(uhval - uval)*(uhval - uval)
        e *= mesh.area()
        return np.sqrt(e.sum()) 


#        e = (self.uh - self.uI)**2
#        area = V.smspace.area
#        e = e[cell]/np.repeat(NV, NV)
#        e = np.bincount(np.repeat(range(NC), NV), weights=e, minlength=NC)*area
#        return np.sqrt(np.sum(e)) 

    def H1_error(self):
        pass

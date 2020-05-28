import numpy as np

from .function import Function

class RTDof3d:
    def __init__(self, mesh, p):
        """
        Parameters
        ----------
        mesh : TetrahedronMesh or HalfEdgeMesh3d object
        p : the space order, p>=0

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.p = p # 默认的空间次数 p >= 0
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def boundary_dof(self, threshold=None):
        idx = self.mesh.ds.boundary_face_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('face', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        face2dof = self.face_to_dof()
        isBdDof[face2dof[idx]] = True
        return isBdDof

    def face_to_dof(self):
        p = self.p
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs('face') 
        face2dof = np.arange(NF*fdof).reshape(NF, fdof)
        return face2dof

    def cell_to_dof(self):
        """
        """
        p = self.p 
        mesh = self.mesh
        if p == 0:
            cell2face = mesh.ds.cell_to_face()
            return cell2face
        else:
            fdof = self.number_of_local_dofs('face') 
            cdof = self.number_of_local_dofs('cell')
            NC = mesh.number_of_cells()
            cell2dof = np.zeros((NC, cdof), dtype=np.int)

            face2dof = self.face_to_dof()
            face2cell = mesh.ds.face_to_cell()

            cell2dof[face2cell[:, [0]], face2cell[:, [2]]*fdof + np.arange(fdof)] = face2dof
            cell2dof[face2cell[:, [1]], face2cell[:, [3]]*fdof + np.arange(fdof)] = face2dof

            idof = ldof - 4*fdof 
            cell2dof[:, 4*fdof:] = NF*fdof+ np.arange(NC*idof).reshape(NC, idof)
            return cell2dof

    def number_of_local_dofs(self, etype='cell'):
        p = self.p
        if etype == 'cell':
            return (p+1)*(p+2)*(p+4)//2 
        elif etype == 'face':
            return (p+1)*(p+2)//2

    def number_of_global_dofs(self):
        p = self.p

        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs('face') 
        gdof = NF*fdof
        if p > 0:
            cdof = self.number_of_local_dofs('cell')
            idof = cdof - 4*fdof
            NC = self.mesh.number_of_cells()
            gdof += NC*ldof
        return gdof 

class RaviartThomasFiniteElementSpace3d:
    def __init__(self, mesh, p=0, q=None):
        self.p = p
        self.mesh = mesh

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator


    def basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()

        ldof = self.number_of_local_dofs()

        p = self.p
        phi = np.zeors((NC, ldof, dim), dtype=self.dtype)


        return phi

    def grad_basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        p = self.p

        gradPhi = np.zeros((NC, ldof, dim, dim), dtype=self.dtype)

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        divPhi = np.zeors((NC, ldof), dtype=self.dtype)

        return divPhi 

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 0:
            return NE
        elif p==1:
            return 2*NE
        else:
            #TODO: raise a error
            print("error!")


    def number_of_local_dofs(self):
        p = self.p
        if p==0:
            return 3
        elif p==1:
            return 6
        else:
            #TODO: raise a error
            print("error!")

    # helper function for understand RT finite element  
    def show_face_frame(self, axes, index):
        pass

    def show_basis(self, axes, index):
        pass


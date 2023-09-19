import numpy as np
from numpy.linalg import inv

from .Function import Function
from .scaled_monomial_space_2d import ScaledMonomialSpace2d
from .vector_monomial_space_2d import VectorMonomialSpace2d 
class CVVEDof2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.itype = self.mesh.itype
        self.cell2dof = self.cell_to_dof()
    def is_boundary_dof(self, threshold=None):
        mesh = self.mesh
        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        idx = mesh.ds.boundary_edge_flag()

        if threshold is not None:
            bc = self.mesh.entity_barycenter('edge')
            idx = threshold(bc)
        isBdDof[edge2dof[idx]] = True   
        return isBdDof
    def edge_to_dof(self, index=np.s_[:]):
        e2p = self.mesh.edge_to_ipoint(self.p, index=index)
        NE, ldof = e2p.shape
        edge2dof = np.zeros((NE, 2*ldof), dtype=e2p.dtype)
        edge2dof[:, 0::2] = 2*e2p
        edge2dof[:, 1::2] = edge2dof[:, 0::2] + 1
        return edge2dof

    face_to_dof = edge_to_dof

    def cell_to_dof(self, index=np.s_[:]):
        """
        """
        p = self.p
        cell = self.mesh.ds._cell

        if p == 1:
            cellLocation = self.mesh.ds.cellLocation
            cell2dof = np.zeros(2*len(cell), dtype=cell.dtype)
            cell2dof[0::2] = 2*cell
            cell2dof[1::2] = cell2dof[0::2] + 1
            return np.hsplit(cell2dof, 2*cellLocation[1:-1])[index]

        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='all')

        location = np.zeros(NC+1, dtype=self.itype)
        location[1:] = np.add.accumulate(ldof)

        cell2dof = np.zeros(location[-1], dtype=self.itype)

        edge2dof = self.edge_to_dof()
        edge2cell = self.mesh.ds.edge_to_cell()

        edof = self.number_of_local_dofs(doftype='edge')
        idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*(edof-2)+ np.arange(edof-2)
        cell2dof[idx] = edge2dof[:, 0:edof-2]

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*(edof-2)).reshape(-1, 1) + np.arange(edof-2) 
        idx1 = (np.arange(-2,-edof,-2).reshape(-1,1)+np.array([0,1])).flatten()
        cell2dof[idx] = edge2dof[isInEdge][:, idx1]

        NN = self.mesh.number_of_nodes()
        NV = self.mesh.ds.number_of_vertices_of_cells()
        NE = self.mesh.number_of_edges()
        cdof = self.number_of_local_dofs(doftype='cell') 
        idx = (location[:-1] + NV*2*p).reshape(-1, 1) + np.arange(cdof)
        cell2dof[idx] = 2*NN + NE*2*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)
        return np.hsplit(cell2dof, location[1:-1])[index]

    def number_of_global_dofs(self):
        return 2*self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='all'):
        return 2*self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def interpolation_points(self, index=np.s_[:], scale:float=0.3):
        p = self.p
        NE = self.mesh.number_of_edges()
        GD = self.mesh.geo_dimension()
        NC = self.mesh.number_of_cells()

        egdof = self.mesh.number_of_nodes()+NE*(p-1)
        ipoint = np.zeros((egdof+p*(p-1)*NC,GD),dtype=np.float_)

        if p==1:
            return self.mesh.interpolation_points(self.p, scale=0.3)[:egdof,:]

        ipoint[:egdof, :] = self.mesh.interpolation_points(self.p, scale=0.3)[:egdof,:]
        bc = self.mesh.entity_barycenter('cell')
        h = np.sqrt(self.mesh.cell_area())[:, None]*scale
        t = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]], dtype=np.float_)
        t1 = np.array([
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.0, 0.0]], dtype=np.float_)

        if p==2:
            ipoint[egdof:,:] = (bc[:, None, :]+t1[None, :2, :]*h[:, :, None]).reshape(-1,GD)   
            return ipoint

        bcs2 = (self.mesh.multi_index_matrix(p-1, etype=2)/(p-1))[:-1, :]
        tri1 = bc[:, None, :]+t[None, :, :]*h[:, :, None]
        tri2 = bc[:, None, :]+t1[None, :, :]*h[:, :, None]

        if p == 3:
            ipoint[egdof:egdof+(p-2)*(p-1)//2*NC, :] =  bc
            ipoint[egdof+(p-2)*(p-1)//2*NC:,:]= np.einsum('ij,kjm->kim', bcs2, tri2).reshape(-1, GD)
            return ipoint

        bcs1 = self.mesh.multi_index_matrix(p-3, etype=2)/(p-3)
        ipoint[egdof:egdof+(p-2)*(p-1)//2*NC, :] = np.einsum('ij,kjm->kim',
                bcs1, tri1).reshape(-1, GD)
        ipoint[egdof+(p-2)*(p-1)//2*NC:, :]= np.einsum('ij,kjm->kim',
                bcs2, tri2).reshape(-1, GD)
        return ipoint
       
        

class ConformingVectorVESpace2d:
    def __init__(self, mesh, p=1, q=None, bc=None):
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q, bc=bc)
        self.vmspace = VectorMonomialSpace2d(mesh, p, q=q, bc=bc)

        self.itype = mesh.itype
        self.ftype = mesh.ftype
        self.dof = CVVEDof2d(mesh, p)


    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell2dof[index]

    def interpolation_points(self, index=np.s_[:]):
        return self.dof.interpolation_points()

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)


    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, coordtype='cartesian', dtype=dtype)



    def edge_basis(self, x, x0):
        """
        @brief 虚单元基函数在边上lagrange插值
        @param x0 是插值点
        @param x 是已知点,此处值为1 其余为0
        @example qf = GaussLobattoQuadrature(p + 2) # NQ
            qf0 = GaussLobattoQuadrature(p+1)
            phi = space.edge_basis(qf0.quadpts[:, 0], qf.quadpts[:, 0])
        """

        A = x[:, None] - x[None, :]
        A[np.arange(len(A)), np.arange(len(A))] = 1.0
        A = np.prod(A, axis=1)
        val = np.zeros((len(x0), len(x)), dtype=np.float64)
        for i in range(len(x)):
            flag = np.ones(len(x), dtype=np.bool_)
            flag[i] = False
            val[:, i] = np.prod(x0[:, None] - x[flag][None, :], axis=1)/A[i]
        return val

    def project_to_vmspace(self, uh, PI1):
        """
        Project a conforming vem function uh into polynomial space.
        """
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape)
        p = self.p
        g = lambda x: x[0]@uh[x[1]]
        S = self.vmspace.function(dim=dim)
        S[:] = np.concatenate(list(map(g, zip(PI1, cell2dof))))
        return S





        
 


import numpy as np

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace.Function import Function
from fealpy.quadrature import FEMeshIntegralAlg

from fealpy.functionspace.femdof import multi_index_matrix2d

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.decorator import cartesian, barycentric

class NedelecDof():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = multi_index_matrix2d(p)

    def is_normal_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isndof = np.zeros(ldof, dtype=np.bool_)

        multiindex = self.multiindex
        dim = np.sum(multiindex>0, axis=-1)-1
        dim0, = np.where(dim==0)
        dim1, = np.where(dim==1)

        isndof[dim0] = True
        isndof[dim1] = True
        isndof[dim0+ldof//2] = True
        return isndof

    def edge_to_local_dof(self):
        multiindex = self.multiindex
        ldof = self.number_of_local_dofs()

        eldof = self.number_of_local_dofs('edge')
        e2ld = np.zeros((3, eldof), dtype=np.int_)

        e2ld[0], = np.where(multiindex[:, 0]==0)
        e2ld[0][0] += ldof//2
        e2ld[0][-1] += ldof//2

        e2ld[1] = np.where(multiindex[:, 1]==0)[0][::-1]
        e2ld[1][-1] += ldof//2

        e2ld[2], = np.where(multiindex[:, 2]==0)
        return e2ld

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+2) 
        elif doftype in {'cell', 2}: # number of dofs inside the cell 
            return (p-1)*(p-2) + 3*(p-1)
        elif doftype in {'face', 'edge', 1}: # number of dofs on each edge 
            return p+1
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        cdof = self.number_of_local_dofs(doftype='cell')
        return NE*edof + NC*cdof

    def edge_to_dof(self, index=np.s_[:]):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        return np.arange(NE*edof).reshape(NE, edof)[index]

    def cell_to_dof(self):
        p = self.p
        cdof = self.number_of_local_dofs('cell')
        edof = self.number_of_local_dofs('edge')
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()
        e2ldof = self.edge_to_local_dof()
        isndof = self.is_normal_dof()

        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        c2e = self.mesh.ds.cell_to_edge()
        c2esign = self.mesh.ds.cell_to_edge_sign()

        c2d = np.zeros((NC, ldof), dtype=np.int_)
        # c2d[:, e2ldof] : (NC, 3, p+1), e2dof[c2e] : (NC, 3, p+1)
        tmp = e2dof[c2e]
        tmp[~c2esign] = tmp[~c2esign, ::-1]
        c2d[:, e2ldof] = tmp 
        c2d[:, ~isndof] = np.arange(NE*edof, gdof).reshape(NC, -1)
        return c2d

    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        eidx = self.mesh.ds.boundary_edge_index()
        e2d = self.edge_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        bddof = self.boundary_dof()

        gdof = self.number_of_global_dofs()
        flag = np.zeros(gdof, dtype=np.bool_)

        flag[bddof] = True
        return flag

class SecondNedelecFiniteElementSpace2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.dof = NedelecDof(mesh, p)

        self.lspace = LagrangeFiniteElementSpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(mesh, p+3, cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

    @barycentric
    def basis(self, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        c2v = self.basis_vector()#(NC, ldof, GD)
        
        shape = bc.shape[:-1]
        val = np.zeros(shape+(NC, ldof, GD), dtype=np.float_)

        bval = self.lspace.basis(bc) #(NQ, NC, ldof)
        c2v = np.broadcast_to(c2v, val.shape)
        val[..., :ldof//2, :] = bval[..., None]*c2v[..., :ldof//2, :]
        val[..., ldof//2:, :] = bval[..., None]*c2v[..., ldof//2:, :]
        return val

    def basis_vector(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        e2ldof = self.dof.edge_to_local_dof()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        c2e = mesh.ds.cell_to_edge()
        e2n = mesh.edge_unit_normal()
        e2t = mesh.edge_unit_tangent()

        c2v = np.zeros((NC, ldof, GD), dtype=np.float_)
        c2v[:, :ldof//2, 0] = 1
        c2v[:, ldof//2:, 1] = 1

        #c2v[:, e2ldof[:, 1:-1]] : (NC, 3, p-1, GD), e2t[c2e] : (NC, 3, GD)
        c2v[:, e2ldof[:, 1:-1]] = e2t[c2e][:, :, None, :]
        c2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[c2e][:, :, None, :]

        #c2v[:, e2ldof[0, 0]] : (NC, GD), c2e[e2t[0]] : (NC, GD)
        #注意: e2ldof[0, 0] 不是 0
        c2v[:, e2ldof[0, 0]] = e2n[c2e[:, 2]]/(np.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 0]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[0, -1]] = e2n[c2e[:, 1]]/(np.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 0]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[1, 0]] = e2n[c2e[:, 0]]/(np.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 1]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[1, -1]] = e2n[c2e[:, 2]]/(np.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 1]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[2, 0]] = e2n[c2e[:, 1]]/(np.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 2]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[2, -1]] = e2n[c2e[:, 0]]/(np.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 2]],
            axis=-1)[:, None]) 
        return c2v

    def dof_vector(self):
        pass

    def edge_basis_vector(self):
        return self.edge_dof_vector()

    def edge_dof_vector(self, index=np.s_[:]):
        """
        @brief 获取每条边上每个自由度处的dof方向 
        @return (NE, p+1, 2)
        """
        #TODO : 测试

        NE = self.mesh.number_of_edges()
        GD = self.mesh.geo_dimension()
        p = self.p

        e = np.arange(NE)[index]
        N = len(e)
        e2dv = np.zeros([N, p+1, GD], dtype=np.int_)
        e2dv[:] = self.mesh.edge_unit_tangent(index=index)[:, None]
        return e2dv

    @barycentric
    def edge_basis(self, bc, index=np.s_[:]):
        mesh = self.mesh
        GD = mesh.geo_dimension()
        edof = self.dof.number_of_local_dofs('edge')
        sphi = self.lspace.basis(bc, index=index) #(NQ, NE, edof)
        e2t = mesh.edge_unit_tangent()

        val = sphi[..., :, None]*e2t[None, index, None, :]
        return val

    @barycentric
    def curl_basis(self, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        node = mesh.entity("node")
        cell = mesh.entity("cell")

        c2d = self.dof.cell_to_dof()
        c2v = self.basis_vector()#(NC, ldof, GD)
        c2e = mesh.ds.cell_to_edge()
        e2n = mesh.edge_unit_normal()
        e2t = mesh.edge_unit_tangent()
        
        sgval = self.lspace.grad_basis(bc) #(NQ, NC, ldof//2, GD)
        val = np.zeros(sgval.shape[:-2]+(ldof,), dtype=np.float_)

        val[..., :ldof//2] = np.cross(sgval, c2v[None, ..., :ldof//2, :])
        val[..., ldof//2:] = np.cross(sgval, c2v[None, ..., ldof//2:, :])
        return val

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = np.einsum("cl, ...clk->...ck", uh[c2d], phi)
        return val

    @barycentric
    def curl_value(self, uh, bc, index=np.s_[:]):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.curl_basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = np.einsum("cl, ...cl->...c", uh[c2d], phi)
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def edge_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def face_value(self, uh, bc, index=np.s_[:]):
        pass

    def mass_matrix(self, c = 1):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, ldof, GD)

        if callable(c): 
            points = mesh.bc_to_point(bcs) #(NQ, NC, GD)
            cval = c(points)               #(NQ, NC)
            mass = np.einsum("qc, qclg, qcdg, c, q->cld", cval, phi, phi, cm, ws)
        else:
            mass = c*np.einsum("qclg, qcdg, c, q->cld", phi, phi, cm, ws)

        I = np.broadcast_to(c2d[:, :, None], shape=mass.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=mass.shape)
        M = csr_matrix((mass.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def curl_matrix(self, c=1):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure

        c2d = self.dof.cell_to_dof() #(NC, ldof)
        bcs, ws = self.integrator.get_quadrature_points_and_weights()

        cphi = self.curl_basis(bcs) #(NQ, NC, ldof)

        if callable(c): 
            points = mesh.bc_to_point(bcs) #(NQ, NC, GD)
            cval = c(points)               #(NQ, NC)
            A = np.einsum("qc, qcl, qcd, c, q->cld", cval, cphi, cphi, cm, ws) #(NC, ldof, ldof)
        else:
            A = c*np.einsum("qcl, qcd, c, q->cld", cphi, cphi, cm, ws) #(NC, ldof, ldof)

        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return B

    def source_vector(self, f):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        fval = f(p) #(NQ, NC, GD)

        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        val = np.einsum("qcg, qclg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = np.zeros(gdof, dtype=np.float_)
        np.add.at(vec, c2d, val)
        return vec

    def projection(self, f, method="L2"):
        M = self.mass_matrix()
        b = self.source_vector(f)
        x = spsolve(M, b)
        return self.function(array=x)

    def function(self, dim=None, array=None, dtype=np.float_):
        if array is None:
            gdof = self.dof.number_of_global_dofs()
            array = np.zeros(gdof, dtype=np.float_)
        return Function(self, dim=dim, array=array, coordtype='barycentric', dtype=dtype)

    def interplation(self, f):
        mesh = self.mesh
        node = mesh.entity("node")
        edge = mesh.entity("edge")

        gdof = self.dof.number_of_global_dofs()
        e2n = mesh.edge_unit_normal()
        val = np.zeros(gdof, dtype=np.float_)

        f0 = f(node[edge[:, 0]]) 
        f1 = f(node[edge[:, 1]])

        val[0::2] = np.sum(f0*e2n, axis=1)
        val[1::2] = np.sum(f1*e2n, axis=1)
        return self.function(array=val)

    def L2_error(self, u, uh):
        '''@
        @brief 计算 ||u - uh||_{L_2}
        '''
        mesh = self.mesh
        cm = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        uval = u(p) #(NQ, NC, GD)
        uhval = uh(bcs) #(NQ, NC, GD)
        errval = np.sum((uval-uhval)*(uval-uhval), axis=-1)#(NQ, NC)
        val = np.einsum("qc, q, c->", errval, ws, cm)
        return np.sqrt(val)

    def set_neumann_bc(self, g):
        bcs, ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()

        edof = self.dof.number_of_local_dofs('edge')
        eidx = self.mesh.ds.boundary_edge_index()
        phi = self.edge_basis(bcs, index=eidx) #(NQ, NE0, edof, GD)
        e2n = self.mesh.edge_unit_normal(index=eidx)
        phi = np.einsum("qelg, eg->qel", phi, e2n) #(NQ, NE0, edof)

        point = self.mesh.edge_bc_to_point(bcs, index=eidx)
        gval = g(point) #(NQ, NE0)

        em = self.mesh.entity_measure("edge")[eidx]
        integ = np.einsum("qel, qe, e, q->el", phi, gval, em, ws)

        e2d = np.ones((len(eidx), edof), dtype=np.int_)
        e2d[:, 0] = edof*eidx
        e2d = np.cumsum(e2d, axis=-1)

        gdof = self.dof.number_of_global_dofs()
        val = np.zeros(gdof, dtype=np.float_)
        np.add.at(val, e2d, integ)
        return val

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        p = self.p
        mesh = self.mesh
        ldof = p+1
        gdof = self.number_of_global_dofs()
       
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()

        edge2dof = self.dof.edge_to_dof()[index]
        e2v = self.edge_dof_vector(index=index) #(NE, p+1, 2)

        bcs = self.lspace.multi_index_matrix[1](p)/p
        point = mesh.bc_to_point(bcs, index=index)#(p+1, NE, GD)

        t = mesh.edge_unit_tangent(index=index)
        gval = gD(point, t)[..., None]*t

        uh[edge2dof] = np.sum(gval.swapaxes(0, 1)*e2v, axis=-1)

        isDDof = np.zeros(gdof, dtype=np.bool_)
        isDDof[edge2dof] = True
        return isDDof



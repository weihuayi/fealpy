
from typing import Union, TypeVar, Generic, Callable,Optional

from ..backend import TensorLike
from ..backend import backend_manager as bm
from .space import FunctionSpace
from .lagrange_fe_space import LagrangeFESpace
from .function import Function

from scipy.sparse import csr_matrix
from ..mesh.mesh_base import Mesh
from ..decorator import barycentric, cartesian

_MT = TypeVar('_MT', bound=Mesh)    
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)


class BDMDof():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = mesh.multi_index_matrix(p, 2) 
        self.ftype = mesh.ftype 
        self.itype = mesh.itype

    def is_normal_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isndof = bm.zeros(ldof, dtype=bm.bool)

        multiindex = self.multiindex
        dim = bm.sum(multiindex>0, axis=-1)-1
        dim0, = bm.where(dim==0)
        dim1, = bm.where(dim==1)

        isndof[dim0] = True
        isndof[dim1] = True
        isndof[dim0+ldof//2] = True
        return isndof

    def edge_to_local_dof(self):
        multiindex = self.multiindex
        ldof = self.number_of_local_dofs()

        eldof = self.number_of_local_dofs('edge')
        e2ld = bm.zeros((3, eldof), dtype=self.itype)

        e2ld[0], = bm.where(multiindex[:, 0]==0)
        e2ld[0][0] += ldof//2
        e2ld[0][-1] += ldof//2

        #e2ld[1] = bm.where(multiindex[:, 1]==0)[0][::-1]
        array = bm.where(multiindex[:, 1]==0)[0]
        e2ld[1] = bm.flip(array)
        e2ld[1][-1] += ldof//2

        e2ld[2], = bm.where(multiindex[:, 2]==0)
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

    def edge_to_dof(self, index=_S):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        return bm.arange(NE*edof).reshape(NE, edof)[index]

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
        c2e = self.mesh.cell_to_edge()
        c2esign = self.mesh.cell_to_face_sign()

        c2d = bm.zeros((NC, ldof), dtype=bm.int64)
        # c2d[:, e2ldof] : (NC, 3, p+1), e2dof[c2e] : (NC, 3, p+1)
        tmp = e2dof[c2e]
        # tmp[~c2esign] = tmp[~c2esign, ::-1]       
        # c2d[:, e2ldof] = tmp
        # c2d[:, ~isndof] = bm.arange(NE*edof, gdof).reshape(NC, -1)
        tmp[~c2esign] = bm.flip(tmp[~c2esign], [1])
        c2d = bm.set_at(c2d,(slice(None), e2ldof),tmp)
        c2d = bm.set_at(c2d,(slice(None), ~isndof),bm.arange(NE*edof, gdof).reshape(NC, -1))
        return c2d

    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        eidx = self.mesh.boundary_face_index()
        e2d = self.edge_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        bddof = self.boundary_dof()

        gdof = self.number_of_global_dofs()
        flag = bm.zeros(gdof, dtype=bm.bool)

        flag[bddof] = True
        return flag
    
class BrezziDouglasMariniFESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p, space=None):
        self.p = p
        self.mesh = mesh
        self.dof = BDMDof(mesh, p)

        self.lspace = LagrangeFESpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.qf = mesh.quadrature_formula(p+3)
        self.ftype = mesh.ftype
        self.itype = mesh.itype

        #TODO:JAX
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()


    @barycentric
    def basis(self, bcs: TensorLike,index=_S)-> TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        node = mesh.entity("node")
        cell = mesh.entity("cell")

        c2v = self.basis_vector()#(NC, ldof, GD)
        
        shape = bcs.shape[:-1]
        val = bm.zeros((NC,)+shape+(ldof, GD), device=self.device, dtype=self.ftype)

        bval = self.lspace.basis(bcs) #(NC, NQ, ldof)
        c2v = c2v[:,None,:,:]
        c2v = bm.broadcast_to(c2v, val.shape)

        # val[..., :ldof//2, :] = bval[..., None]*c2v[..., :ldof//2, :]
        # val[..., ldof//2:, :] = bval[..., None]*c2v[..., ldof//2:, :]
        val = bm.set_at(val,(..., slice(None, ldof//2), slice(None)),bval[..., None]*c2v[..., :ldof//2, :])
        val = bm.set_at(val,(..., slice(ldof//2, None), slice(None)),bval[..., None]*c2v[..., ldof//2:, :])
        return val[index]


    def basis_vector(self)->TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        e2ldof = self.dof.edge_to_local_dof()

        c2e = mesh.cell_to_edge()
        e2n = mesh.edge_unit_normal()
        e2t = mesh.edge_unit_tangent()

        c2v = bm.zeros((NC, ldof, GD), device=self.device, dtype=self.ftype)
        # c2v[:, :ldof//2, 0] = 1
        # c2v[:, ldof//2:, 1] = 1
        c2v = bm.set_at(c2v,(slice(None), slice(None, ldof//2), 0),1)
        c2v = bm.set_at(c2v,(slice(None), slice(ldof//2, None), 1),1)

        #c2v[:, e2ldof[:, 1:-1]] : (NC, 3, p-1, GD), e2t[c2e] : (NC, 3, GD)
        # c2v[:, e2ldof[:, 1:-1]] = e2n[c2e][:, :, None, :]
        # c2v[:, e2ldof[:, 1:-1]+ldof//2] = e2t[c2e][:, :, None, :]
        
        c2v = bm.set_at(c2v,(slice(None), e2ldof[:, 1:-1]),e2n[c2e][:, :, None, :])
        c2v = bm.set_at(c2v,(slice(None), e2ldof[:, 1:-1]+ldof//2),e2t[c2e][:, :, None, :])


        #c2v[:, e2ldof[0, 0]] : (NC, GD), c2e[e2t[0]] : (NC, GD)
        # c2v[:, e2ldof[0, 0]] = e2t[c2e[:, 2]]/(bm.sum(e2t[c2e[:, 2]]*e2n[c2e[:, 0]],axis=-1)[:, None]) 
        # c2v[:, e2ldof[0, -1]] = e2t[c2e[:, 1]]/(bm.sum(e2t[c2e[:, 1]]*e2n[c2e[:, 0]],axis=-1)[:, None]) 
        # c2v[:, e2ldof[1, 0]] = e2t[c2e[:, 0]]/(bm.sum(e2t[c2e[:, 0]]*e2n[c2e[:, 1]],axis=-1)[:, None]) 
        # c2v[:, e2ldof[1, -1]] = e2t[c2e[:, 2]]/(bm.sum(e2t[c2e[:, 2]]*e2n[c2e[:, 1]],axis=-1)[:, None]) 
        # c2v[:, e2ldof[2, 0]] = e2t[c2e[:, 1]]/(bm.sum(e2t[c2e[:, 1]]*e2n[c2e[:, 2]],axis=-1)[:, None]) 
        # c2v[:, e2ldof[2, -1]] = e2t[c2e[:, 0]]/(bm.sum(e2t[c2e[:, 0]]*e2n[c2e[:, 2]],axis=-1)[:, None]) 
        
        c2v = bm.set_at(c2v,(slice(None), e2ldof[0, 0]),e2t[c2e[:, 2]]/(bm.sum(e2t[c2e[:, 2]]*e2n[c2e[:, 0]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None), e2ldof[0, -1]),e2t[c2e[:, 1]]/(bm.sum(e2t[c2e[:, 1]]*e2n[c2e[:, 0]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None), e2ldof[1, 0]),e2t[c2e[:, 0]]/(bm.sum(e2t[c2e[:, 0]]*e2n[c2e[:, 1]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None), e2ldof[1, -1]),e2t[c2e[:, 2]]/(bm.sum(e2t[c2e[:, 2]]*e2n[c2e[:, 1]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None), e2ldof[2, 0]),e2t[c2e[:, 1]]/(bm.sum(e2t[c2e[:, 1]]*e2n[c2e[:, 2]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None), e2ldof[2, -1]),e2t[c2e[:, 0]]/(bm.sum(e2t[c2e[:, 0]]*e2n[c2e[:, 2]],axis=-1)[:, None]))
        
        return c2v

    @barycentric
    def edge_basis(self, bcs: TensorLike, index : Index = _S)->TensorLike:
        mesh = self.mesh
        GD = mesh.geo_dimension()
        edof = self.dof.number_of_local_dofs('edge')
        sphi = self.lspace.basis(bcs, index=index) #(NQ, NE, edof)

        e2n = mesh.edge_unit_normal()
        val = sphi[..., None]*e2n[index, None, None, :]

        return val

    @barycentric
    def div_basis(self, bcs: TensorLike,index=_S)->TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        node = mesh.entity("node")
        cell = mesh.entity("cell")

        c2v = self.basis_vector()#(NC, ldof, GD)
        
        shape = bcs.shape[:-1]
        val = bm.zeros((NC,)+shape+(ldof,),device=self.device, dtype=self.ftype)

        sgval = self.lspace.grad_basis(bcs) #(NQ, NC, ldof, GD)
        c2v = c2v[:,None,:,:]
        c2v = bm.broadcast_to(c2v, val.shape+(GD,))
        # val[..., :ldof//2] = bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., :ldof//2, :])
        # val[..., ldof//2:] = bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., ldof//2:, :])
        val = bm.set_at(val,(..., slice(None, ldof//2)),bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., :ldof//2, :]))
        val = bm.set_at(val,(..., slice(ldof//2, None)),bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., ldof//2:, :]))
        return val[index]
    
    def edge_basis_vector(self):
        return self.edge_dof_vector()

    def edge_dof_vector(self, index : Index =_S)->TensorLike:
        """
        @brief 获取每条边上每个自由度处的dof方向 
        @return (NE, p+1, 2)
        """
        #TODO : 测试

        NE = self.mesh.number_of_edges()
        GD = self.mesh.geo_dimension()
        p = self.p

        e = bm.arange(NE)[index]
        N = len(e)
        e2dv = bm.zeros([N, p+1, GD], dtype=self.ftype)
        #e2dv[:] = self.mesh.edge_unit_normal(index=index)[:, None]
        e2dv = bm.set_at(e2dv,(slice(None)),self.mesh.edge_unit_normal(index=index)[:, None])
        return e2dv

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)
    
    def is_boundary_dof(self, threshold=None,method=None):
        return self.dof.is_boundary_dof()

    @barycentric
    def value(self, uh, bcs, index=_S):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.basis(bcs)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = bm.einsum("cl, cqlk->cqk", uh[c2d], phi)
        return val

    @barycentric
    def div_value(self, uh, bcs, index=_S):
        pass

    @barycentric
    def grad_value(self, uh, bc, index=_S):
        pass

    @barycentric
    def edge_value(self, uh, bc, index=_S):
        pass

    @barycentric
    def face_value(self, uh, bc, index=_S):
        pass

    def mass_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.qf.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NC, NQ, ldof, GD)
        mass = bm.einsum("cqlg, cqdg, c, q->cld", phi, phi, cm, ws)

        I = bm.broadcast_to(c2d[:, :, None], shape=mass.shape)
        J = bm.broadcast_to(c2d[:, None, :], shape=mass.shape)
        M = csr_matrix((mass.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def div_matrix(self, space):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof0 = self.dof.number_of_global_dofs()
        gdof1 = space.dof.number_of_global_dofs()
        cm = self.cellmeasure

        c2d = self.dof.cell_to_dof() #(NC, ldof)
        c2d_space = space.dof.cell_to_dof()

        bcs, ws = self.qf.get_quadrature_points_and_weights()

        #if space.basis.coordtype == 'barycentric':
        fval = space.basis(bcs) #(NQ, NC, ldof1)
        #else:
        #    points = self.mesh.bc_to_point(bcs)
        #    fval = space.basis(points)

        phi = self.div_basis(bcs) #(NQ, NC, ldof)
        A = bm.einsum("cql, cqd, c, q->cld", phi, fval, cm, ws)

        I = bm.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = bm.broadcast_to(c2d_space[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
        return B

    def source_vector(self, f):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        p = mesh.bc_to_point(bcs) #(NC, NQ, GD)
        fval = f(p) #(NC, NQ, GD)

        phi = self.basis(bcs) #(NC, NQ, ldof, GD)
        val = bm.einsum("cqg, cqlg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = bm.zeros(gdof,device=self.device, dtype=self.ftype)
        bm.add.at(vec, c2d, val)
        return vec

#     def projection(self, f, method="L2"):
#         M = self.mass_matrix()
#         b = self.source_vector(f)
#         x = spsolve(M, b)
#         return self.function(array=x)

#     def function(self, dim=None, array=None, dtype=np.float_):
#         if array is None:
#             gdof = self.dof.number_of_global_dofs()
#             array = np.zeros(gdof, dtype=np.float_)
#         return Function(self, dim=dim, array=array, coordtype='barycentric', dtype=dtype)

#     def interplation(self, f):
#         pass

#     def L2_error(self, u, uh):
#         '''@
#         @brief 计算 ||u - uh||_{L_2}
#         '''
#         mesh = self.mesh
#         cm = self.cellmeasure
#         bcs, ws = self.integrator.get_quadrature_points_and_weights()
#         p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
#         uval = u(p) #(NQ, NC, GD)
#         uhval = uh(bcs) #(NQ, NC, GD)
#         errval = np.sum((uval-uhval)*(uval-uhval), axis=-1)#(NQ, NC)
#         val = np.einsum("qc, q, c->", errval, ws, cm)
#         return np.sqrt(val)

#     def set_neumann_bc(self, g):
#         bcs, ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()

#         edof = self.dof.number_of_local_dofs('edge')
#         eidx = self.mesh.ds.boundary_edge_index()
#         phi = self.edge_basis(bcs, index=eidx) #(NQ, NE0, edof, GD)
#         e2n = self.mesh.edge_unit_normal(index=eidx)
#         phi = np.einsum("qelg, eg->qel", phi, e2n) #(NQ, NE0, edof)

#         point = self.mesh.edge_bc_to_point(bcs, index=eidx)
#         gval = g(point) #(NQ, NE0)

#         em = self.mesh.entity_measure("edge")[eidx]
#         integ = np.einsum("qel, qe, e, q->el", phi, gval, em, ws)

#         e2d = np.ones((len(eidx), edof), dtype=np.int_)
#         e2d[:, 0] = edof*eidx
#         e2d = np.cumsum(e2d, axis=-1)

#         gdof = self.dof.number_of_global_dofs()
#         val = np.zeros(gdof, dtype=np.float_)
#         np.add.at(val, e2d, integ)
#         return val

    def set_neumann_bc(self, g):
        p = self.p
        mesh = self.mesh

        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        edof = self.dof.number_of_local_dofs('edge')
        eidx = self.mesh.boundary_face_index()
        gdof = self.dof.number_of_global_dofs()
        edge2dof = self.dof.edge_to_dof()[eidx]

        phi = self.edge_basis(bcs)[eidx] #(NE, NQ, edof, GD)
        e2n = self.mesh.edge_unit_normal(index=eidx)
        phi = bm.einsum("eqlg, eg->eql", phi, e2n) #(NE, NQ, edof)

        points = mesh.bc_to_point(bcs)[eidx]
        gval = g(points) 

        em = self.mesh.entity_measure("edge")[eidx]
        vec = bm.zeros(gdof, device=self.device, dtype=self.ftype)
        vec[edge2dof] = bm.einsum("eql, eq, e, q->el", phi, gval, em, ws)

        return vec
    
    def set_dirichlet_bc(self, gd, uh, threshold=None, q=None,method=None):
        p = self.p
        mesh = self.mesh
        ldof = p+1
        gdof = self.number_of_global_dofs()
       
        if type(threshold) is bm.array:
            index = threshold
        else:
            index = self.mesh.boundary_face_index()

        edge2dof = self.dof.edge_to_dof()[index]
        e2v = self.edge_dof_vector(index=index) #(NE, p+1, 3)

        bcs = self.mesh.multi_index_matrix(p, 1, dtype=self.ftype)/p
        point = mesh.bc_to_point(bcs, index=index)
        gval = gd(point,e2v) #(NE, p+1, 3)
        #uh[edge2dof] = bm.sum(gval*e2v, axis=-1)
        uh[edge2dof] = gval

        isDDof = bm.zeros(gdof, dtype=bm.bool)
        isDDof[edge2dof] = True
        return uh,isDDof

    boundary_interpolate = set_dirichlet_bc
    

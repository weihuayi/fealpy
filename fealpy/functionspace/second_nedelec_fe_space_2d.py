
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

class NedelecDof():
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

class SecondNedelecFESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh : Mesh, p: int):
        self.p = p
        self.mesh = mesh
        self.dof = NedelecDof(mesh, p)

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

        c2v = self.basis_vector()#(NC, ldof, GD)
        
        NQ = bcs.shape[0]
        val = bm.zeros((NC, NQ, ldof, GD), dtype=self.ftype)

        bval = self.lspace.basis(bcs) #(NC, NQ, ldof)
        c2v = c2v[:,None,:,:]
        c2v = bm.broadcast_to(c2v, val.shape) #(NC, NQ, ldof, GD)
        #val[..., :ldof//2, :] = bval[..., None]*c2v[..., :ldof//2, :]       
        #val[..., ldof//2:, :] = bval[..., None]*c2v[..., ldof//2:, :]
        val = bm.set_at(val,(...,slice(None,ldof//2),slice(None)),bval[..., None]*c2v[..., :ldof//2, :])
        val = bm.set_at(val,(...,slice(ldof//2,None),slice(None)),bval[..., None]*c2v[..., ldof//2:, :])
        return val[index]

    def basis_vector(self)->TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        e2ldof = self.dof.edge_to_local_dof()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        c2e = mesh.cell_to_edge()
        e2n = mesh.edge_unit_normal()
        e2t = mesh.edge_unit_tangent()

        c2v = bm.zeros((NC, ldof, GD), dtype=self.ftype)
        #c2v[:, :ldof//2, 0] = 1        
        #c2v[:, ldof//2:, 1] = 1
        # 给内部单元的单元向量赋值
        c2v = bm.set_at(c2v,(slice(None),slice(None,ldof//2),0),1)
        c2v = bm.set_at(c2v,(slice(None),slice(ldof//2,None),1),1)

        #c2v[:, e2ldof[:, 1:-1]] : (NC, 3, p-1, GD), e2t[c2e] : (NC, 3, GD)
        #c2v[:, e2ldof[:, 1:-1]] = e2t[c2e][:, :, None, :]
        #c2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[c2e][:, :, None, :]
        # 给边界单元(不包含顶点)的单元向量赋值
        c2v = bm.set_at(c2v,(slice(None),e2ldof[:, 1:-1]),e2t[c2e][:, :, None, :])
        c2v = bm.set_at(c2v,(slice(None),e2ldof[:, 1:-1]+ldof//2),e2n[c2e][:, :, None, :])


        #c2v[:, e2ldof[0, 0]] : (NC, GD), c2e[e2t[0]] : (NC, GD)
        #注意: e2ldof[0, 0] 不是 0

        # c2v[:, e2ldof[0, 0]] = e2n[c2e[:, 2]]/(bm.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 0]],
        #     axis=-1)[:, None]) 
        # c2v[:, e2ldof[0, -1]] = e2n[c2e[:, 1]]/(bm.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 0]],
        #     axis=-1)[:, None]) 
        # c2v[:, e2ldof[1, 0]] = e2n[c2e[:, 0]]/(bm.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 1]],
        #     axis=-1)[:, None]) 
        # c2v[:, e2ldof[1, -1]] = e2n[c2e[:, 2]]/(bm.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 1]],
        #     axis=-1)[:, None]) 
        # c2v[:, e2ldof[2, 0]] = e2n[c2e[:, 1]]/(bm.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 2]],
        #     axis=-1)[:, None]) 
        # c2v[:, e2ldof[2, -1]] = e2n[c2e[:, 0]]/(bm.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 2]],
        #     axis=-1)[:, None])
      # 给顶点单元的单元向量赋值
        c2v = bm.set_at(c2v,(slice(None),e2ldof[0, 0]),e2n[c2e[:, 2]]/(bm.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 0]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None),e2ldof[0, -1]),e2n[c2e[:, 1]]/(bm.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 0]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None),e2ldof[1, 0]),e2n[c2e[:, 0]]/(bm.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 1]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None),e2ldof[1, -1]),e2n[c2e[:, 2]]/(bm.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 1]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None),e2ldof[2, 0]),e2n[c2e[:, 1]]/(bm.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 2]],axis=-1)[:, None]))
        c2v = bm.set_at(c2v,(slice(None),e2ldof[2, -1]),e2n[c2e[:, 0]]/(bm.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 2]],axis=-1)[:, None]))

        return c2v

    def dof_vector(self):
        pass

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
        #e2dv[:] = self.mesh.edge_unit_tangent(index=index)[:, None]
        e2dv = bm.set_at(e2dv,(slice(None)),self.mesh.edge_unit_tangent(index=index)[:, None])
        return e2dv

    @barycentric
    def edge_basis(self, bcs: TensorLike, index : Index = _S)->TensorLike:
        """
        @brief 计算每条边上每个重心坐标处的基函数值
        @param bc : (NQ, GD+1)
        @return val : (NE, NQ, edof, GD)
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        edof = self.dof.number_of_local_dofs('edge')

        sphi = self.lspace.basis(bcs, index=index) #(NE, NQ, edof)
        e2t = mesh.edge_unit_tangent() #(NE, GD)

        val = sphi[..., None]*e2t[index, None, None, :]
        return val
    
    def cross2d(self,a,b):
        return a[...,0]*b[...,1] - a[...,1]*b[...,0]

    @barycentric
    def curl_basis(self, bcs: TensorLike)->TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        node = mesh.entity("node")
        cell = mesh.entity("cell")

        c2d = self.dof.cell_to_dof()
        c2v = self.basis_vector()#(NC, ldof, GD)
        c2e = mesh.cell_to_edge()
        e2n = mesh.edge_unit_normal()
        e2t = mesh.edge_unit_tangent()
        
        sgval = self.lspace.grad_basis(bcs) #(NC, NQ, ldof//2, GD)
        val = bm.zeros(sgval.shape[:-2]+(ldof,), dtype=self.ftype)

        # val[..., :ldof//2] = bm.cross(sgval, c2v[:, None, :ldof//2, :])
        # val[..., ldof//2:] = bm.cross(sgval, c2v[:, None, ldof//2:, :])
        val = bm.set_at(val,(...,slice(None,ldof//2)),self.cross2d(sgval, c2v[:, None, :ldof//2, :]))
        val = bm.set_at(val,(...,slice(ldof//2,None)),self.cross2d(sgval, c2v[:, None, ldof//2:, :]))

        return val

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)
    
    def is_boundary_dof(self, threshold=None,method=None):
        return self.dof.is_boundary_dof()

    @barycentric
    def value(self, uh, bcs: TensorLike, index: Index = _S)->TensorLike:
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (NQ, TD+1)
        @return val :  (NC, NQ, GD)
        '''
        phi = self.basis(bcs)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (NC, NQ, ldof, GD)
        val = bm.einsum("cl, cqlk->cqk", uh[c2d], phi)
        return val

    @barycentric
    def curl_value(self, uh, bcs: TensorLike, index: Index = _S)->TensorLike:
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.curl_basis(bcs)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = bm.einsum("cl, cql->cq", uh[c2d], phi)
        return val

    @barycentric
    def div_value(self, uh, bc: TensorLike, index: Index = _S)->TensorLike:
        pass

    @barycentric
    def grad_value(self, uh, bc: TensorLike, index: Index = _S)->TensorLike:
        pass

    @barycentric
    def edge_value(self, uh, bc: TensorLike, index: Index = _S)->TensorLike:
        pass

    @barycentric
    def face_value(self, uh, bc: TensorLike, index: Index = _S)->TensorLike:
        pass

    def mass_matrix(self, c: Union[float, Callable]=1):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.qf.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        if callable(c): 
            points = mesh.bc_to_point(bcs) #(NC, NQ, GD)
            cval = c(points)               #(NC, NQ)
            mass = bm.einsum("cq, cqlg, cqmg, c, q->clm", cval, phi, phi, cm, ws)
        else:
            mass = c*bm.einsum("cqlg, cqmg, c, q->clm", phi, phi, cm, ws)

        I = bm.broadcast_to(c2d[:, :, None], shape=mass.shape)
        J = bm.broadcast_to(c2d[:, None, :], shape=mass.shape)
        M = csr_matrix((mass.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def curl_matrix(self, c: Union[float, Callable]=1):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure

        c2d = self.dof.cell_to_dof() #(NC, ldof)
        bcs, ws = self.qf.get_quadrature_points_and_weights()

        cphi = self.curl_basis(bcs) #(NQ, NC, ldof)

        if callable(c): 
            points = mesh.bc_to_point(bcs) #(NQ, NC, GD)
            cval = c(points)               #(NQ, NC)
            A = bm.einsum("cq, cql, cqm, c, q->clm", cval, cphi, cphi, cm, ws) #(NC, ldof, ldof)
        else:
            A = c*bm.einsum("cql, cqm, c, q->clm", cphi, cphi, cm, ws) #(NC, ldof, ldof)

        I = bm.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = bm.broadcast_to(c2d[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return B

    def source_vector(self, f: Callable):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        fval = f(p) #(NQ, NC, GD)

        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        val = bm.einsum("cqg, cqlg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = bm.zeros(gdof, dtype=self.ftype)
        #bm.scatter_add(vec, c2d, val)
        bm.add_at(vec, c2d, val)
        return vec

    # def projection(self, f:Callable, method="L2"):
    #     M = self.mass_matrix()
    #     b = self.source_vector(f)
    #     x = spsolve(M, b)
    #     return x 

    def interplation(self, f: Callable):
        mesh = self.mesh
        node = mesh.entity("node")
        edge = mesh.entity("edge")

        gdof = self.dof.number_of_global_dofs()
        e2n = mesh.edge_unit_normal()
        val = bm.zeros(gdof, dtype=self.ftype)

        f0 = f(node[edge[:, 0]]) 
        f1 = f(node[edge[:, 1]])

        val[0::2] = bm.sum(f0*e2n, axis=1)
        val[1::2] = bm.sum(f1*e2n, axis=1)
        return val 

    def L2_error(self, u: Callable, uh: Callable):
        '''@
        @brief 计算 ||u - uh||_{L_2}
        '''
        mesh = self.mesh
        cm = self.cellmeasure
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        uval = u(p) #(NQ, NC, GD)
        uhval = uh(bcs) #(NQ, NC, GD)
        errval = bm.sum((uval-uhval)*(uval-uhval), axis=-1)#(NQ, NC)
        val = bm.einsum("cq, q, c->", errval, ws, cm)
        return bm.sqrt(val)

    def set_neumann_bc(self,gD):
        p = self.p
        mesh = self.mesh
        isbdFace = mesh.boundary_face_flag()
        edge2dof = self.dof.edge_to_dof()[isbdFace]
        fm = mesh.entity_measure('face')[isbdFace]
        gdof = self.number_of_global_dofs()
       
        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = self.edge_basis(bcs,index=isbdFace)
        points = mesh.bc_to_point(bcs)[isbdFace]
        t = mesh.edge_unit_tangent()[isbdFace]
        hval = gD(points,t)
        vec = bm.zeros(gdof, dtype=self.ftype)
        vec[edge2dof] = bm.einsum('eqg, eqlg,q,e->el', hval, bphi,ws,fm) # (NE, ldof)
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



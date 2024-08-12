

from typing import Union, TypeVar, Generic, Callable, Optional
from ..typing import TensorLike, Index, _S

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from ...decorator import barycentric
from .space import FunctionSpace

class NedelecDof(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = bm.multi_index_matrix(p, 2) 

    def is_normal_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isndof = bm.zeros(ldof, dtype=bm.bool_)

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
        e2ld = bm.zeros((3, eldof), dtype=bm.int_)

        e2ld[0], = bm.where(multiindex[:, 0]==0)
        e2ld[0][0] += ldof//2
        e2ld[0][-1] += ldof//2

        e2ld[1] = bm.where(multiindex[:, 1]==0)[0][::-1]
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

    def edge_to_dof(self, index=bm.s_[:]):
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
        c2e = self.mesh.ds.cell_to_edge()
        c2esign = self.mesh.ds.cell_to_edge_sign()

        c2d = bm.zeros((NC, ldof), dtype=bm.int_)
        # c2d[:, e2ldof] : (NC, 3, p+1), e2dof[c2e] : (NC, 3, p+1)
        tmp = e2dof[c2e]
        tmp[~c2esign] = tmp[~c2esign, ::-1]
        c2d[:, e2ldof] = tmp 
        c2d[:, ~isndof] = bm.arange(NE*edof, gdof).reshape(NC, -1)
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
        flag = bm.zeros(gdof, dtype=bm.bool_)

        flag[bddof] = True
        return flag

class SecondNedelecFiniteElementSpace2d():
    def __init__(self, mesh : Mesh, p: int):
        self.p = p
        self.mesh = mesh
        self.dof = NedelecDof(mesh, p)

        self.lspace = LagrangeFiniteElementSpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.qf = mesh.quadrature_formula(p+3) 

    @barycentric
    def basis(self, bc: TensorLike)-> TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        c2v = self.basis_vector()#(NC, ldof, GD)
        
        NQ = bc.shape[0]
        val = bm.zeros((NC, NQ, ldof, GD), dtype=bm.float_)

        bval = self.lspace.basis(bc) #(NC, NQ, ldof)
        c2v = bm.broadcast_to(c2v, val.shape) #(NC, NQ, ldof, GD)
        val[..., :ldof//2, :] = bval[..., None]*c2v[..., :ldof//2, :]
        val[..., ldof//2:, :] = bval[..., None]*c2v[..., ldof//2:, :]
        return val

    def basis_vector(self)->TensorLike:
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

        c2v = bm.zeros((NC, ldof, GD), dtype=bm.float_)
        c2v[:, :ldof//2, 0] = 1
        c2v[:, ldof//2:, 1] = 1

        #c2v[:, e2ldof[:, 1:-1]] : (NC, 3, p-1, GD), e2t[c2e] : (NC, 3, GD)
        c2v[:, e2ldof[:, 1:-1]] = e2t[c2e][:, :, None, :]
        c2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[c2e][:, :, None, :]

        #c2v[:, e2ldof[0, 0]] : (NC, GD), c2e[e2t[0]] : (NC, GD)
        #注意: e2ldof[0, 0] 不是 0
        c2v[:, e2ldof[0, 0]] = e2n[c2e[:, 2]]/(bm.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 0]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[0, -1]] = e2n[c2e[:, 1]]/(bm.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 0]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[1, 0]] = e2n[c2e[:, 0]]/(bm.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 1]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[1, -1]] = e2n[c2e[:, 2]]/(bm.sum(e2n[c2e[:, 2]]*e2t[c2e[:, 1]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[2, 0]] = e2n[c2e[:, 1]]/(bm.sum(e2n[c2e[:, 1]]*e2t[c2e[:, 2]],
            axis=-1)[:, None]) 
        c2v[:, e2ldof[2, -1]] = e2n[c2e[:, 0]]/(bm.sum(e2n[c2e[:, 0]]*e2t[c2e[:, 2]],
            axis=-1)[:, None]) 
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
        e2dv = bm.zeros([N, p+1, GD], dtype=bm.int_)
        e2dv[:] = self.mesh.edge_unit_tangent(index=index)[:, None]
        return e2dv

    @barycentric
    def edge_basis(self, bc: TensorLike, index : Index = _S)->TensorLike:
        """
        @brief 计算每条边上每个重心坐标处的基函数值
        @param bc : (NQ, GD+1)
        @return val : (NE, NQ, edof, GD)
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        edof = self.dof.number_of_local_dofs('edge')

        sphi = self.lspace.basis(bc, index=index) #(NE, NQ, edof)
        e2t = mesh.edge_unit_tangent() #(NE, GD)

        val = sphi[..., None]*e2t[index, None, None, :]
        return val

    @barycentric
    def curl_basis(self, bc: TensorLike)->TensorLike:
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
        
        sgval = self.lspace.grad_basis(bc) #(NC, NQ, ldof//2, GD)
        val = bm.zeros(sgval.shape[:-2]+(ldof,), dtype=bm.float_)

        val[..., :ldof//2] = bm.cross(sgval, c2v[:, None, :ldof//2, :])
        val[..., ldof//2:] = bm.cross(sgval, c2v[:, None, ldof//2:, :])
        return val

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bc: TensorLike, index: Index = _S)->TensorLike:
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (NQ, TD+1)
        @return val :  (NC, NQ, GD)
        '''
        phi = self.basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (NC, NQ, ldof, GD)
        val = bm.einsum("cl, cqlk->cqk", uh[c2d], phi)
        return val

    @barycentric
    def curl_value(self, uh, bc: TensorLike, index: Index = _S)->TensorLike:
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.curl_basis(bc)
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
        bcs, ws = self.integrator.get_quadrature_points_and_weights()

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
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        fval = f(p) #(NQ, NC, GD)

        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        val = bm.einsum("cqg, cqlg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = bm.zeros(gdof, dtype=bm.float_)
        bm.scatter_add(vec, c2d, val)
        return vec

    def projection(self, f:Callable, method="L2"):
        M = self.mass_matrix()
        b = self.source_vector(f)
        x = spsolve(M, b)
        return x 

    def interplation(self, f: Callable):
        mesh = self.mesh
        node = mesh.entity("node")
        edge = mesh.entity("edge")

        gdof = self.dof.number_of_global_dofs()
        e2n = mesh.edge_unit_normal()
        val = bm.zeros(gdof, dtype=bm.float_)

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
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        uval = u(p) #(NQ, NC, GD)
        uhval = uh(bcs) #(NQ, NC, GD)
        errval = bm.sum((uval-uhval)*(uval-uhval), axis=-1)#(NQ, NC)
        val = bm.einsum("cq, q, c->", errval, ws, cm)
        return bm.sqrt(val)

    def set_neumann_bc(self, g: Callable):
        bcs, ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()

        edof = self.dof.number_of_local_dofs('edge')
        eidx = self.mesh.ds.boundary_edge_index()
        phi = self.edge_basis(bcs, index=eidx) #(NQ, NE0, edof, GD)
        e2n = self.mesh.edge_unit_normal(index=eidx)
        phi = bm.einsum("eqlg, eg->eql", phi, e2n) #(NE0, NQ, edof)

        point = self.mesh.edge_bc_to_point(bcs, index=eidx)
        gval = g(point) #(NE0, NQ)

        em = self.mesh.entity_measure("edge")[eidx]
        integ = bm.einsum("eql, eq, e, q->el", phi, gval, em, ws)

        e2d = bm.ones((len(eidx), edof), dtype=bm.int_)
        e2d[:, 0] = edof*eidx
        e2d = bm.cumsum(e2d, axis=-1)

        gdof = self.dof.number_of_global_dofs()
        val = bm.zeros(gdof, dtype=bm.float_)
        bm.scatter_add(val, e2d, integ) #TODO : 有问题 scatter_add
        return val

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        p = self.p
        mesh = self.mesh
        ldof = p+1
        gdof = self.number_of_global_dofs()
       
        if type(threshold) is bm.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()

        edge2dof = self.dof.edge_to_dof()[index]
        e2v = self.edge_dof_vector(index=index) #(NE, p+1, 2)

        bcs = self.lspace.multi_index_matrix[1](p)/p
        point = mesh.bc_to_point(bcs, index=index)#(NE, p+1, GD)

        t = mesh.edge_unit_tangent(index=index)
        gval = gD(point, t)[..., None]*t

        uh[edge2dof] = bm.sum(gval.swapaxes(0, 1)*e2v, axis=-1)

        isDDof = bm.zeros(gdof, dtype=bm.bool_)
        isDDof[edge2dof] = True
        return isDDof



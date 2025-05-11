
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
        self.mesh = mesh
        self.p = p
        self.multiindex2 = mesh.multi_index_matrix(p,2)
        self.multiindex3 = mesh.multi_index_matrix(p,3)
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device=mesh.device


    def edge_to_local_face_dof(self):
        multiindex = self.multiindex2
        ldof = self.number_of_local_dofs()

        e2ld = bm.zeros((3, self.p+1),device=self.device,  dtype=self.itype)

        e2ld[0], = bm.where(multiindex[:, 0]==0)
        
        #e2ld[1] = bm.where(multiindex[:, 1]==0)[0][::-1]
        array = bm.where(multiindex[:, 1]==0)[0]
        e2ld[1] =  bm.flip(array)
        
        e2ld[2], = bm.where(multiindex[:, 2]==0)
        return e2ld

    def cell_to_dof(self):
        face = self.mesh.entity("face")
        cell = self.mesh.entity("cell")

        multiindex = self.multiindex2.T #(3, ldof)
        cell2face = self.mesh.cell_to_face()
        localFace = bm.array([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)],device=self.device, dtype=self.itype)

        f2d = self.face_to_dof()
        f2ld = self.face_to_local_dof()
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        cdof = self.number_of_local_dofs('cell')
        fdof = self.number_of_local_dofs('face')

        isndof = bm.zeros(ldof,device=self.device, dtype=bm.bool)
        isndof[f2ld] = True

        c2d = bm.zeros((NC, ldof),device=self.device, dtype=self.itype)
        idx = bm.zeros((NC, 3),device=self.device, dtype=self.itype)
        for i in range(4):
            fi = face[cell2face[:, i]] #(NC, 3)
            fj = cell[:, localFace[i]]
            _, idx[:, 0] = bm.where(fj==fi[:, 0, None])
            _, idx[:, 1] = bm.where(fj==fi[:, 1, None])
            _, idx[:, 2] = bm.where(fj==fi[:, 2, None])
            k = multiindex[idx[:, 1]] + multiindex[idx[:, 2]] #(NC, fdof)
            didx = k*(k+1)//2+multiindex[idx[:, 2]]
            #c2d[:, f2ld[i]] = f2d[cell2face[:, [i]], didx]
            c2d = bm.set_at(c2d,(slice(None), f2ld[i]), f2d[cell2face[:, [i]], didx])
        #c2d[:, ~isndof] = bm.arange(NF*fdof, gdof).reshape(NC, cdof)
        c2d = bm.set_at(c2d,(slice(None), ~isndof), bm.arange(NF*fdof, gdof).reshape(NC, cdof))
        return c2d
    
    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def face_to_local_dof(self):
        multiindex = self.multiindex3
        ldof = self.number_of_local_dofs()

        fdof = self.number_of_local_dofs('face')
        f2ld = bm.zeros((4, fdof),device=self.device, dtype=self.itype)
        eldof = self.edge_to_local_face_dof()

        f2ld[0], = bm.where(multiindex[:, 0]==0)
        # f2ld[0, eldof[:, 1:-1]] += ldof//3 # 底面编号最大
        # f2ld[0, eldof[:, 0]] += 2*(ldof//3)
        f2ld = bm.set_at(f2ld,(0, eldof[:, 1:-1]), f2ld[0, eldof[:, 1:-1]]+ldof//3)
        f2ld = bm.set_at(f2ld,(0, eldof[:, 0]), f2ld[0, eldof[:, 0]]+2*(ldof//3))

        f2ld[1], = bm.where(multiindex[:, 1]==0)
        # f2ld[1, eldof[[1, 2], :]] += ldof//3 # 底面编号最大
        # f2ld[1, eldof[1, -1]] += ldof//3
        f2ld = bm.set_at(f2ld,(1, eldof[[1, 2], :]), f2ld[1, eldof[[1, 2], :]]+ldof//3)
        f2ld = bm.set_at(f2ld,(1, eldof[1, -1]), f2ld[1, eldof[1, -1]]+ldof//3)

        f2ld[2], = bm.where(multiindex[:, 2]==0)
        # f2ld[2, eldof[2, :]] += ldof//3 # 底面编号最大
        f2ld = bm.set_at(f2ld,(2, eldof[2, :]), f2ld[2, eldof[2, :]]+ldof//3)

        f2ld[3], = bm.where(multiindex[:, 3]==0)
        return f2ld

    def face_to_dof(self, index=_S):
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face')
        return bm.arange(NF*fdof,device=self.device, dtype=self.itype).reshape(NF, fdof)[index]

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+2)*(p+3)//2
        elif doftype in {'cell'}: # number of dofs inside the cell 
            return ((p+1)*(p+2)*(p+3)//2) - 2*(p+1)*(p+2)
        elif doftype in {'face'}: # number of dofs on each edge 
            return (p+1)*(p+2)//2
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face')
        cdof = self.number_of_local_dofs(doftype='cell')
        return NF*fdof + NC*cdof

    def is_boundary_dof(self):

        gdof = self.number_of_global_dofs()             
        index = self.mesh.boundary_face_index()
        face2dof = self.face_to_dof()[index]

        isDDof = bm.zeros(gdof, dtype=bm.bool)
        isDDof[face2dof] = True
        return isDDof

class BrezziDouglasMariniFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p, space=None, q = None):
        
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
        val = bm.zeros((NC,)+shape+(ldof, GD),device=self.device, dtype=self.ftype)

        bval = self.lspace.basis(bcs) #(NQ, NC, ldof//3)
        c2v = c2v[:,None,:,:]
        c2v = bm.broadcast_to(c2v, val.shape)

        # val[..., :ldof//3, :] = bval[..., None]*c2v[..., :ldof//3, :]
        # val[..., ldof//3 : 2*(ldof//3):, :] = bval[..., None]*c2v[..., ldof//3:2*(ldof//3), :]
        # val[..., 2*(ldof//3):, :] = bval[..., None]*c2v[..., 2*(ldof//3):, :]
        val =bm.set_at(val,(...,slice(None,ldof//3),slice(None)), bval[..., None]*c2v[..., :ldof//3, :])
        val =bm.set_at(val,(...,slice(ldof//3,2*(ldof//3)),slice(None)), bval[..., None]*c2v[..., ldof//3:2*(ldof//3), :])  
        val =bm.set_at(val,(...,slice(2*(ldof//3),None),slice(None)), bval[..., None]*c2v[..., 2*(ldof//3):, :])  
        
        return val[index]

    @barycentric
    def face_basis(self, bcs, index=_S):
        mesh = self.mesh
        GD = mesh.geo_dimension()
        fdof = self.dof.number_of_local_dofs('face')
        sphi = self.lspace.basis(bcs, index=index) #(NQ, NF, edof)

        f2n = mesh.face_unit_normal()
        val = sphi[..., None]*f2n[index,None, None, :]
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
        val = bm.zeros((NC,)+shape+(ldof,), dtype=self.ftype, device=self.device)

        sgval = self.lspace.grad_basis(bcs) #(NQ, NC, ldof, GD)
        c2v = c2v[:,None,:,:]
        c2v = bm.broadcast_to(c2v, val.shape+(GD,))
        # val[..., :ldof//3] = bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., :ldof//3, :])
        # val[..., ldof//3:2*(ldof//3)] = bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., ldof//3:2*(ldof//3), :])
        # val[..., -ldof//3:] = bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., -ldof//3:, :])
        val = bm.set_at(val,(...,slice(None,ldof//3)), bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., :ldof//3, :]))
        val = bm.set_at(val,(...,slice(ldof//3,2*(ldof//3))), bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., ldof//3:2*(ldof//3), :]))
        val = bm.set_at(val,(...,slice(2*(ldof//3),None)), bm.einsum('ijkl, ijkl->ijk', sgval, c2v[..., 2*(ldof//3):, :]))
        
        return val[index]

    def basis_vector(self):

        NC = self.mesh.number_of_cells()

        c2e = self.mesh.cell_to_edge()
        c2f = self.mesh.cell_to_face()
        f2e = self.mesh.face_to_edge()
        e2t = self.mesh.edge_tangent()
        em = self.mesh.entity_measure("edge")
        e2t = e2t/em[:, None]
        f2n = self.mesh.face_unit_normal()

        lf2f = bm.tensor([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)],device=self.device, dtype=self.itype) 
        lf2e = bm.tensor([[5, 4, 3], [5, 2, 1], [4, 2, 0], [3, 1, 0]],device=self.device, dtype=self.itype) #(4, 3)

        e2fdof = self.dof.edge_to_local_face_dof() #(3, p+1) 面上, 在边界的自由度 
        f2ldof = self.dof.face_to_local_dof() # 单元上, 在面上的自由度
        ldof = self.dof.number_of_local_dofs() 
        fdof = self.dof.number_of_local_dofs('face')

        bv = bm.zeros((NC, ldof, 3), device=self.device, dtype=self.ftype)

        # 内部向量
        # bv[:, :ldof//3, 0] = 1
        # bv[:, ldof//3:2*(ldof//3), 1] = 1
        # bv[:, -ldof//3:, 2] = 1

        bv = bm.set_at(bv,(slice(None), slice(None,ldof//3), 0), 1)
        bv = bm.set_at(bv,(slice(None), slice(ldof//3,2*(ldof//3)), 1), 1)
        bv = bm.set_at(bv,(slice(None), slice(-ldof//3,None), 2), 1)

        # 面上的内部自由度
        isfndof = bm.ones(fdof, dtype=bm.bool)
        isfndof[e2fdof] = False

        # 面上的法向与切向
        # bv[:, f2d] : (NC, 4, fdof, 3), f2n[c2f] : (NC, 4, 3), c2e[:, lf2e[:, 1]] : (NC, 4)
        # bv[:, f2ldof[:, isfndof]] = f2n[c2f, None]
        # bv[:, f2ldof[:, isfndof]+ldof//3] = e2t[c2e[:, lf2e[:, 0]], None]
        # bv[:, f2ldof[:, isfndof]+2*(ldof//3)] = bm.linalg.cross(f2n[c2f, None], e2t[c2e[:, lf2e[:, 0]], None])
        
        bv = bm.set_at(bv,(slice(None), f2ldof[:, isfndof]), f2n[c2f, None])
        bv = bm.set_at(bv,(slice(None), f2ldof[:, isfndof]+ldof//3), e2t[c2e[:, lf2e[:, 0]], None])
        bv = bm.set_at(bv,(slice(None), f2ldof[:, isfndof]+2*(ldof//3)), bm.linalg.cross(f2n[c2f, None], e2t[c2e[:, lf2e[:, 0]], None]))

        #边上的法向
        # bv[:, f2ldof[:, eldof]] (NC, 4, 3, eldof, 3)
        # e2t[c2e[:, lf2e], None] : (NC, 4, 3, 1, 3), f2n[c2f[:, lf2f], None] : (NC, 4, 3, 1, 3)
        tmp = bm.linalg.cross(e2t[c2e[:, lf2e], None], f2n[c2f[:, lf2f], None])
        #bv[:, f2ldof[:, e2fdof[:, 1:-1]]] = tmp/bm.sum(tmp*f2n[c2f, None, None], axis=-1)[..., None]
        bv = bm.set_at(bv,(slice(None), f2ldof[:, e2fdof[:, 1:-1]]), tmp/bm.sum(tmp*f2n[c2f, None,None], axis=-1)[..., None])

        # 边上的切向
        # bv[:, f2ldof[3, e2fdof[:, 1:-1]]+2*(ldof//3)] = e2t[c2e[:, lf2e[3]], None]
        # bv[:, f2ldof[2, e2fdof[[0, 1], 1:-1]]+2*(ldof//3)] = e2t[c2e[:, lf2e[2, [0, 1]]], None]
        # bv[:, f2ldof[1, e2fdof[[0], 1:-1]]+2*(ldof//3)] = e2t[c2e[:, lf2e[1, [0]]], None]
        
        bv = bm.set_at(bv,(slice(None), f2ldof[3, e2fdof[:, 1:-1]]+2*(ldof//3)), e2t[c2e[:, lf2e[3]], None])
        bv = bm.set_at(bv,(slice(None), f2ldof[2, e2fdof[[0, 1], 1:-1]]+2*(ldof//3)), e2t[c2e[:, lf2e[2, [0, 1]]], None])
        bv = bm.set_at(bv,(slice(None), f2ldof[1, e2fdof[[0], 1:-1]]+2*(ldof//3)), e2t[c2e[:, lf2e[1, [0]]], None])
       
        # 面的三个顶点连接的, 不再这个面上的边
        lf2se = bm.tensor([[1, 2, 0], [3, 4, 0], [3, 5, 1], [4, 5, 2]],device=self.device, dtype=self.itype) #(4, 3)

        # 顶点上的法向
        tmp = e2t[c2e[:, lf2se]] #(NC, 4, 3, 3)
        # bv[:, f2ldof[:, e2fdof[:, 0]]] = tmp/bm.sum(tmp*f2n[c2f, None], axis=-1)[..., None]
        bv = bm.set_at(bv,(slice(None), f2ldof[:, e2fdof[:, 0]]), tmp/bm.sum(tmp*f2n[c2f, None], axis=-1)[..., None])
        return bv

    def dof_vector(self):
        bv = self.basis_vector()
        NC = bv.shape[0]
        dv = bm.linalg.inv(bv.reshape(NC, 3, -1, 3).swapaxes(1, 2)).swapaxes(-1,
                -2).swapaxes(1, 2).reshape(NC, -1, 3)
        return dv

    def mass_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.qf.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
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

        # if space.basis.coordtype == 'barycentric':
        fval = space.basis(bcs) #(NQ, NC, ldof1)
        # else:
        #     points = self.mesh.bc_to_point(bcs)
        #     fval = space.basis(points)

        phi = self.div_basis(bcs) #(NQ, NC, ldof)
        A = bm.einsum("cql, cqd, c, q->cld", phi, fval, cm, ws)

        I = bm.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = bm.broadcast_to(c2d_space[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
        return B

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

    def source_vector(self, f):
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
        vec = bm.zeros(gdof, dtype=self.ftype, device=self.device)
        bm.add.at(vec, c2d, val)
        return vec
    
    def set_neumann_bc(self, g):
        p = self.p
        mesh = self.mesh

        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        fidx = self.mesh.boundary_face_index()
        gdof = self.dof.number_of_global_dofs()
        face2dof = self.dof.face_to_dof()[fidx]

        phi = self.face_basis(bcs)[fidx] #(NF, NQ, fdof, GD)
        f2n = self.mesh.face_unit_normal(index=fidx)
        phi = bm.einsum("fqlg, fg->fql", phi, f2n) #(NF, NQ, fdof)

        points = mesh.bc_to_point(bcs)[fidx]
        gval = g(points) 

        fm = self.mesh.entity_measure("face")[fidx]
        vec = bm.zeros(gdof, device=self.device, dtype=self.ftype)
        k= bm.einsum("fql, fq, f, q->fl", phi, gval, fm, ws)
        bm.add.at(vec,face2dof,k)
        return vec

#     def set_neumann_bc(self, g):
#         bcs, ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()

#         fdof = self.dof.number_of_local_dofs('face')
#         fidx = self.mesh.ds.boundary_face_index()
#         phi = self.face_basis(bcs, index=fidx) #(NQ, NE0, edof, GD)
#         f2n = self.mesh.face_unit_normal(index=fidx)
#         phi = np.einsum("qelg, eg->qel", phi, f2n) #(NQ, NE0, edof)

#         point = self.mesh.bc_to_point(bcs, index=fidx)
#         gval = g(point) #(NQ, NE0)

#         fm = self.mesh.entity_measure("face")[fidx]
#         integ = np.einsum("qel, qe, e, q->el", phi, gval, fm, ws)

#         f2d = np.ones((len(fidx), fdof), dtype=np.int_)
#         f2d[:, 0] = fdof*fidx
#         f2d = np.cumsum(f2d, axis=-1)

#         gdof = self.dof.number_of_global_dofs()
#         val = np.zeros(gdof, dtype=np.float_)
#         np.add.at(val, f2d, integ)
#         return val

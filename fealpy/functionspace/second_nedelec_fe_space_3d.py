
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
        self.mesh = mesh
        self.p = p
        self.multiindex2 = mesh.multi_index_matrix(p,2)
        self.multiindex3 = mesh.multi_index_matrix(p,3)
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device=mesh.device

    def edge_to_local_face_dof(self):
        multiindex = self.multiindex2
        ldof = self.number_of_local_dofs("faceall")

        eldof = self.number_of_local_dofs('edge')
        e2ld = bm.zeros((3, eldof), dtype=self.itype)

        e2ld[0], = bm.where(multiindex[:, 0]==0)
        e2ld[0][0] += ldof//2
        e2ld[0][-1] += ldof//2

        #e2ld[1] = bm.where(multiindex[:, 1]==0)[0][::-1]
        array = bm.where(multiindex[:, 1]==0)[0]
        e2ld[1] =  bm.flip(array)

        e2ld[1][-1] += ldof//2

        e2ld[2], = bm.where(multiindex[:, 2]==0)
        return e2ld

    def face_to_local_dof(self):
        multiindex = self.multiindex3
        ldof = self.number_of_local_dofs()

        fdof = self.number_of_local_dofs('faceall')
        f2ld = bm.zeros((4, fdof),device=self.device, dtype=self.itype)
        eldof = self.edge_to_local_face_dof()
        nldof = bm.tensor([[eldof[(i+1)%3, 0], eldof[(i+2)%3, -1]] for i in range(3)])

        localFace = bm.tensor([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)], dtype=self.itype)
        sdof = fdof//2

        for i in range(4):
            flag = localFace[i] > i  
            #f2ld[i, :sdof] = bm.where(multiindex[:, i]==0)[0]           
            #f2ld[i, sdof:] = f2ld[i, :sdof] + ldof//3           
            #f2ld[i, eldof[flag, 1:-1]+sdof] += ldof//3           
            #f2ld[i, nldof[flag]] += ldof//3
            f2ld = bm.set_at(f2ld,(i,slice(None,sdof)),bm.where(multiindex[:, i]==0)[0])
            f2ld = bm.set_at(f2ld,(i,slice(sdof,None)),f2ld[i, :sdof]+ldof//3)
            f2ld = bm.set_at(f2ld, (i, eldof[flag, 1:-1] + sdof), f2ld[i, eldof[flag, 1:-1] + sdof] + ldof // 3)
            f2ld = bm.set_at(f2ld, (i, nldof[flag]), f2ld[i, nldof[flag]] + ldof // 3)
        return f2ld

    def cell_to_dof(self):
        face = self.mesh.entity("face")
        cell = self.mesh.entity("cell")

        multiindex = self.multiindex2.T #(3, ldof)
        cell2face = self.mesh.cell_to_face()
        localFace = bm.tensor([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)],
                dtype=self.itype)

        f2d = self.face_to_dof()
        f2ld = self.face_to_local_dof()
        e2lfd = self.edge_to_local_face_dof()
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        cdof = self.number_of_local_dofs('cell')
        fdof = self.number_of_local_dofs('faceall')

        isndof = bm.zeros(ldof, device=self.device, dtype=bm.bool)
        isndof[f2ld] = True

        c2d = bm.zeros((NC, ldof), device=self.device, dtype=bm.int64)#
        idx = bm.zeros((NC, 3), dtype=self.itype)
        fe = bm.tensor([[0, 1], [0, 2], [1, 2]], dtype=self.itype) #局部边
        for i in range(4):
            fi = face[cell2face[:, i]] #(NC, 3)
            fj = cell[:, localFace[i]]
            _, idx[:, 0] = bm.where(fj==fi[:, 0, None])
            _, idx[:, 1] = bm.where(fj==fi[:, 1, None])
            _, idx[:, 2] = bm.where(fj==fi[:, 2, None])
            k = multiindex[idx[:, 1]] + multiindex[idx[:, 2]] #(NC, fdof)
            didx = k*(k+1)//2+multiindex[idx[:, 2]]
            # c2d[:, f2ld[i, :fdof//2]] = f2d[cell2face[:, [i]], didx]
            # c2d[:, f2ld[i, fdof//2:]] = f2d[cell2face[:, [i]], didx+fdof//2]
            c2d = bm.set_at(c2d,(slice(None),f2ld[i, :fdof//2]),f2d[cell2face[:, [i]], didx])
            c2d = bm.set_at(c2d,(slice(None),f2ld[i, fdof//2:]),f2d[cell2face[:, [i]], didx+fdof//2])

            idx = bm.argsort(idx, axis=1)
            # 顶点的自由度可能需要交换
            for j in range(3):
                flag = bm.sum(idx[:, fe[fe[j, 0]]]-idx[:, fe[fe[j, 1]]], axis=-1)>0
                #tmp = c2d[flag, f2ld[i, e2lfd[(j+1)%3, -1]]].copy()
                tmp = bm.copy(c2d[flag, f2ld[i, e2lfd[(j+1)%3, -1]]])
                # c2d[flag, f2ld[i, e2lfd[(j+1)%3, -1]]] = c2d[flag, f2ld[i, e2lfd[(j+2)%3, 0]]]
                # c2d[flag, f2ld[i, e2lfd[(j+2)%3,qqqqqqqqqqqqqqqqq 0]]] = tmp
                c2d = bm.set_at(c2d,(flag,f2ld[i, e2lfd[(j+1)%3, -1]]),c2d[flag, f2ld[i, e2lfd[(j+2)%3, 0]]])
                c2d = bm.set_at(c2d,(flag,f2ld[i, e2lfd[(j+2)%3, 0]]),tmp)

        # c2d[:, ~isndof] = bm.arange(gdof-NC*cdof, gdof).reshape(NC, cdof)
        c2d = bm.set_at(c2d,(slice(None),~isndof),bm.arange(gdof-NC*cdof, gdof).reshape(NC, cdof))
        return c2d
    
    @property
    def cell2dof(self):
        return self.cell_to_dof()
    
    def face_to_dof(self):
        p = self.p

        edge = self.mesh.entity("edge")
        face = self.mesh.entity("face")

        fdof = self.number_of_local_dofs('face')
        edof = self.number_of_local_dofs('edge')
        ldof = self.number_of_local_dofs('faceall')

        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()
        e2ldof = self.edge_to_local_face_dof()

        istdof = bm.zeros(ldof, dtype=bm.bool)
        istdof[e2ldof] = True

        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        f2e = self.mesh.face_to_edge()

        f2esign = bm.zeros((NF, 3), dtype=bm.bool)
        f2esign = edge[f2e, 0]==face[:, [1, 2, 0]]

        f2d = bm.zeros((NF, ldof), dtype=bm.int64)###
        # c2d[:, e2ldof] : (NC, 3, p+1), e2dof[c2e] : (NC, 3, p+1)
        tmp = e2dof[f2e]
        #tmp[~f2esign] = tmp[~f2esign, ::-1]
        tmp[~f2esign] = bm.flip(tmp[~f2esign], [1])
        # f2d[:, e2ldof] = tmp 
        # f2d[:, ~istdof] = bm.arange(NE*edof, NE*edof+NF*fdof).reshape(NF, -1)
        f2d = bm.set_at(f2d,(slice(None),e2ldof),tmp)
        f2d = bm.set_at(f2d,(slice(None),~istdof),bm.arange(NE*edof, NE*edof+NF*fdof).reshape(NF, -1))

        return f2d

    def edge_to_dof(self):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge')
        return bm.arange(NE*edof).reshape(NE, edof)

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+2)*(p+3)//2 
        elif doftype in {'cell'}: # number of dofs inside the cell 
            return (p-3)*(p-2)*(p-1)//2 + (p-2)*(p-1)*2
        elif doftype in {'faceall'}: # number of all dofs on each face 
            return (p+1)*(p+2)
        elif doftype in {'face'}: # number of dofs on each face 
            return (p-2)*(p-1) + 3*(p-1)
        elif doftype in {'edge'}: # number of dofs on each edge 
            return p+1
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()

        cdof = self.number_of_local_dofs("cell")
        fdof = self.number_of_local_dofs("face")
        edof = self.number_of_local_dofs("edge")
        return NC*cdof+NE*edof+NF*fdof
    
    def is_boundary_dof(self):

        gdof = self.number_of_global_dofs()             
        index = self.mesh.boundary_face_index()
        face2dof = self.face_to_dof()[index]

        isDDof = bm.zeros(gdof, dtype=bm.bool)
        isDDof[face2dof] = True
        return isDDof


class  SecondNedelecFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p, space=None, q = None):
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
    def basis(self, bc,index=_S):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        c2v = self.basis_vector() #(NC, ldof, GD)
        
        shape = bc.shape[:-1]
        val = bm.zeros((NC,)+ shape + (ldof, GD),device=self.device, dtype=self.ftype)

        bval = self.lspace.basis(bc) #(NC, NQ, ldof//3)

        #c2v = bm.broadcast_to(c2v, val.shape)
        # val[..., :ldof//3, :] = bval[..., None]*c2v[..., :ldof//3, :]
        # val[..., ldof//3 : 2*(ldof//3):, :] = bval[..., None]*c2v[..., ldof//3:2*(ldof//3), :]
        # val[..., 2*(ldof//3):, :] = bval[..., None]*c2v[..., 2*(ldof//3):, :]
       
        val = bm.set_at(val,(...,slice(None,ldof//3),slice(None)),bval[..., None]*c2v[:,None, :ldof//3, :])
        val = bm.set_at(val,(...,slice(ldof//3,2*ldof//3),slice(None)),bval[..., None]*c2v[:,None, ldof//3:2*(ldof//3), :])
        val = bm.set_at(val,(...,slice(2*ldof//3,None),slice(None)),bval[..., None]*c2v[:,None, 2*(ldof//3):, :])
        return val[index]


    @barycentric
    def face_basis(self, bc, index=_S):
        mesh = self.mesh
        p = self.p
        GD = mesh.geo_dimension()
        ldof = (p+1)*(p+2) 

        f2v = self.face_basis_vector(index=index)#(NF, ldof, GD)
        NF = len(f2v)
        shape = bc.shape[:-1]
        val = bm.zeros((NF,) + shape+(ldof, GD), device=self.device, dtype=self.ftype)

        bval = self.lspace.basis(bc) #(NF, NQ, ldof//3)

        # f2v = bm.broadcast_to(f2v, val.shape)
        # val[..., :ldof//2, :] = bval[..., None]*f2v[..., :ldof//2, :]
        # val[..., ldof//2:, :] = bval[..., None]*f2v[..., ldof//2:, :]

        val = bm.set_at(val,(...,slice(None,ldof//2),slice(None)),bval[..., None]*f2v[:,None, :ldof//2, :])
        val = bm.set_at(val,(...,slice(ldof//2,None),slice(None)),bval[..., None]*f2v[:,None, ldof//2:, :])
        return val

    def face_basis_vector(self, index=_S):
        p = self.p
        mesh = self.mesh
        GD = mesh.geo_dimension()
        ldof = (p+1)*(p+2) 
        e2ldof = self.dof.edge_to_local_face_dof()

        f2e = mesh.face_to_edge()[index]
        f2n = mesh.face_unit_normal()[index] #(NF, 3)
        #e2t = mesh.edge_unit_tangent()[f2e] #(NF, 3, 3)
        # em = mesh.entity_measure('edge')[f2e]
        # e2t = mesh.edge_tangent()[f2e]/em[:, None]

        em = mesh.entity_measure('edge')
        e2t = mesh.edge_tangent()/em[:, None]
        e2t = e2t[f2e]
      
        e2n = -bm.linalg.cross(f2n[:, None], e2t) #(NF, 3, 3)

        NF = len(f2e)

        f2v = bm.zeros((NF, ldof, GD),device=self.device, dtype=self.ftype)
        # f2v[:, :ldof//2] = e2t[:, 0, None] #(NF, ldof//2, 3)
        # f2v[:, ldof//2:] = e2n[:, 0, None]
        f2v = bm.set_at(f2v,(slice(None),slice(None,ldof//2)),e2t[:, 0, None])
        f2v = bm.set_at(f2v,(slice(None),slice(ldof//2,None)),e2n[:, 0, None])

        #f2v[:, e2ldof[:, 1:-1]] : (NF, 3, p-1, GD), e2t: (NF, 3, 3)
        # f2v[:, e2ldof[:, 1:-1]] = e2t[:, :, None]
        # f2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[:, :, None, :]
        f2v = bm.set_at(f2v,(slice(None),e2ldof[:, 1:-1]),e2t[:, :, None])
        f2v = bm.set_at(f2v,(slice(None),e2ldof[:, 1:-1]+ldof//2),e2n[:, :, None, :])

        #f2v[:, e2ldof[0, 0]] : (NC, GD), f2e[e2t[0]] : (NC, GD)
        #注意: e2ldof[0, 0] 不是 0
        # f2v[:, e2ldof[0, 0]] = e2n[:, 2]/(bm.sum(e2n[:, 2]*e2t[:, 0], axis=-1)[:, None]) 
        # f2v[:, e2ldof[0, -1]] = e2n[:, 1]/(bm.sum(e2n[:, 1]*e2t[:, 0], axis=-1)[:, None]) 
        # f2v[:, e2ldof[1, 0]] = e2n[:, 0]/(bm.sum(e2n[:, 0]*e2t[:, 1], axis=-1)[:, None]) 
        # f2v[:, e2ldof[1, -1]] = e2n[:, 2]/(bm.sum(e2n[:, 2]*e2t[:, 1], axis=-1)[:, None]) 
        # f2v[:, e2ldof[2, 0]] = e2n[:, 1]/(bm.sum(e2n[:, 1]*e2t[:, 2], axis=-1)[:, None]) 
        # f2v[:, e2ldof[2, -1]] = e2n[:, 0]/(bm.sum(e2n[:, 0]*e2t[:, 2], axis=-1)[:, None]) 
        f2v = bm.set_at(f2v,(slice(None),e2ldof[0, 0]),e2n[:, 2]/(bm.sum(e2n[:, 2]*e2t[:, 0], axis=-1)[:, None]))
        f2v = bm.set_at(f2v,(slice(None),e2ldof[0, -1]),e2n[:, 1]/(bm.sum(e2n[:, 1]*e2t[:, 0], axis=-1)[:, None]))
        f2v = bm.set_at(f2v,(slice(None),e2ldof[1, 0]),e2n[:, 0]/(bm.sum(e2n[:, 0]*e2t[:, 1], axis=-1)[:, None]))
        f2v = bm.set_at(f2v,(slice(None),e2ldof[1, -1]),e2n[:, 2]/(bm.sum(e2n[:, 2]*e2t[:, 1], axis=-1)[:, None]))
        f2v = bm.set_at(f2v,(slice(None),e2ldof[2, 0]),e2n[:, 1]/(bm.sum(e2n[:, 1]*e2t[:, 2], axis=-1)[:, None]))
        f2v = bm.set_at(f2v,(slice(None),e2ldof[2, -1]),e2n[:, 0]/(bm.sum(e2n[:, 0]*e2t[:, 2], axis=-1)[:, None]))
        
        return f2v

    def face_dof_vector(self, index=_S):
        p = self.p
        mesh = self.mesh
        GD = mesh.geo_dimension()
        ldof = (p+1)*(p+2) 
        e2ldof = self.dof.edge_to_local_face_dof()

        f2e = mesh.face_to_edge()[index]
        f2n = mesh.face_unit_normal()[index] #(NF, 3)
        #e2t = mesh.edge_unit_tangent()[f2e] #(NF, 3, 3)
        em = mesh.entity_measure('edge')
        e2t = mesh.edge_tangent()/em[:, None]
        e2t = e2t[f2e]
        
        e2n = -bm.linalg.cross(f2n[:, None], e2t) #(NF, 3, 3)

        NF = len(f2e)

        f2v = bm.zeros((NF, ldof, GD), device=self.device,dtype=self.ftype)
        # f2v[:, :ldof//2] = e2t[:, 0, None] #(NF, ldof//2, 3)
        # f2v[:, ldof//2:] = e2n[:, 0, None]
        f2v = bm.set_at(f2v,(slice(None),slice(None,ldof//2)),e2t[:, 0, None])
        f2v = bm.set_at(f2v,(slice(None),slice(ldof//2,None)),e2n[:, 0, None])

        # f2v[:, e2ldof[:, 1:-1]] : (NF, 3, p-1, GD), e2t: (NF, 3, 3)
        # f2v[:, e2ldof] = e2t[:, :, None]
        # f2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[:, :, None, :]
        f2v = bm.set_at(f2v,(slice(None),e2ldof),e2t[:, :, None])
        f2v = bm.set_at(f2v,(slice(None),e2ldof[:, 1:-1]+ldof//2),e2n[:, :, None, :])
        return f2v

    @barycentric
    def curl_basis(self, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()

        node = mesh.entity("node")
        cell = mesh.entity("cell")

        c2v = self.basis_vector()#(NC, ldof, GD)
        sgval = self.lspace.grad_basis(bc) #(NQ, NC, lldof, GD)
        val = bm.zeros((NC,)+(bc.shape[0], )+(ldof, GD),device=self.device, dtype=self.ftype)

        # val[..., :ldof//3, :] = bm.cross(sgval, c2v[:,None, :ldof//3, :])
        # val[..., ldof//3:2*(ldof//3), :] = bm.cross(sgval, c2v[:,None, ldof//3:2*(ldof//3), :])
        # val[..., 2*(ldof//3):, :] = bm.cross(sgval, c2v[:,None, 2*(ldof//3):, :])
           
        val = bm.set_at(val,(...,slice(None,ldof//3),slice(None)),bm.linalg.cross(sgval, c2v[:,None, :ldof//3, :]))
        val = bm.set_at(val,(...,slice(ldof//3,2*ldof//3),slice(None)),bm.linalg.cross(sgval, c2v[:,None, ldof//3:2*(ldof//3), :]))
        val = bm.set_at(val,(...,slice(2*ldof//3,None),slice(None)),bm.linalg.cross(sgval, c2v[:,None, 2*(ldof//3):, :]))

        return val

    def basis_vector(self):
        mesh = self.mesh
        NC = self.mesh.number_of_cells()

        c2e = self.mesh.cell_to_edge()
        c2f = self.mesh.cell_to_face()
        em = mesh.entity_measure('edge')
        e2t = mesh.edge_tangent()/em[:, None]
       
        f2n = self.mesh.face_unit_normal()
        f2e = self.mesh.face_to_edge()

        lf2f = bm.array([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)]) 
        lf2e = bm.array([[5, 4, 3], [5, 2, 1], [4, 2, 0], [3, 1, 0]]) #(4, 3)

        e2fdof = self.dof.edge_to_local_face_dof() #(3, p+1) 面上, 在边界的自由度 
        f2ldof = self.dof.face_to_local_dof() # 单元上, 在面上的自由度
        ldof = self.dof.number_of_local_dofs() 
        fdof = self.dof.number_of_local_dofs('faceall')

        bv = bm.zeros((NC, ldof, 3), dtype=self.ftype)

        # 内部向量
        # bv[:, :ldof//3, 0] = 1
        # bv[:, ldof//3:2*(ldof//3), 1] = 1
        # bv[:, -ldof//3:, 2] = 1
        bv = bm.set_at(bv,(slice(None),slice(None,ldof//3),0),1)
        bv = bm.set_at(bv,(slice(None),slice(ldof//3,2*ldof//3),1),1)
        bv = bm.set_at(bv,(slice(None),slice(-ldof//3,None),2),1)

        # 面内部的标量自由度
        fDofIdx, = bm.where(bm.all(self.dof.multiindex2!=0, axis=-1)) 
        n2fe = bm.array([[(i+1)%3, (i+2)%3] for i in range(3)])
        #n2dof = e2fdof[n2fe[:, ::-1], [[0, -1]]]
        array =  bm.flip(n2fe, axis=1)
        n2dof = e2fdof[array, [[0, -1]]]
        sdof = fdof//2
        for i in range(4):
            # 面内部的自由度
            # bv[:, f2ldof[i, fDofIdx]] = e2t[f2e[c2f[:, i], 0], None] 
            # bv[:, f2ldof[i, fDofIdx+sdof]] = bm.cross(e2t[f2e[c2f[:, i], 0],None], f2n[c2f[:, i], None])
            # bv[:, f2ldof[i, fDofIdx] + 2*(ldof//3)] = f2n[c2f[:, i], None]
            bv = bm.set_at(bv,(slice(None),f2ldof[i, fDofIdx]),e2t[f2e[c2f[:, i], 0], None])
            bv = bm.set_at(bv,(slice(None),f2ldof[i, fDofIdx+sdof]),bm.linalg.cross(e2t[f2e[c2f[:, i], 0],None], f2n[c2f[:, i], None]))
            bv = bm.set_at(bv,(slice(None),f2ldof[i, fDofIdx]+2*(ldof//3)),f2n[c2f[:, i], None])

            #边上的切向自由度
            # bv[:, f2ldof[i, e2fdof[:, 1:-1]]] = e2t[c2e[:, lf2e[i]], None]
            bv = bm.set_at(bv,(slice(None),f2ldof[i, e2fdof[:, 1:-1]]),e2t[c2e[:, lf2e[i]], None])
            # 与边垂直的自由度
            # bv[:, f2ldof[i, e2ldof[:, 1:-1]+sdof]] : (NC, 3, p-1, 3)
            f2fn = f2n[c2f[:, lf2f[i]], None]
            f2ft = bm.linalg.cross(e2t[c2e[:, lf2e[i]], None], f2n[c2f[:, i], None, None])
            #bv[:, f2ldof[i, e2fdof[:, 1:-1]+sdof]] = f2fn/(bm.sum(f2fn*f2ft,axis=-1))[..., None]
            bv = bm.set_at(bv,(slice(None),f2ldof[i, e2fdof[:, 1:-1]+sdof]),f2fn/(bm.sum(f2fn*f2ft,axis=-1))[..., None])
            # 顶点上的自由度
            # bv[:, f2ldof[i, n2dof]] (NC, 3, 2, 3)
            # f2n[c2f[:, n2fe]] (NC, 3, 2, 3)
            tmp0 = f2n[c2f[:, lf2f[i, n2fe]]]
            #tmp1 = e2t[c2e[:, lf2e[i, n2fe[:, ::-1]]]]
            k = bm.flip(n2fe, axis=-1)
            tmp1 = e2t[c2e[:, lf2e[i, k]]]
            #bv[:, f2ldof[i, n2dof]] = tmp0/(bm.sum(tmp0*tmp1, axis=-1))[..., None]
            bv = bm.set_at(bv,(slice(None),f2ldof[i, n2dof]),tmp0/(bm.sum(tmp0*tmp1, axis=-1))[..., None])
        return bv

    def dof_vector(self):
        bv = self.basis_vector()
        NC = bv.shape[0]
        dv = bm.linalg.inv(bv.reshape(NC, 3, -1, 3).swapaxes(1, 2)).swapaxes(-1,
                -2).swapaxes(1, 2).reshape(NC, -1, 3)
        return dv

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def mass_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.qf.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        #mass = bm.einsum("qclg, qcdg, c, q->cld", phi, phi, cm, ws)

        mass = bm.einsum("cqlg, cqmg, c, q->clm", phi, phi, cm, ws)
        I = bm.broadcast_to(c2d[:, :, None], shape=mass.shape)
        J = bm.broadcast_to(c2d[:, None, :], shape=mass.shape)
        M = csr_matrix((mass.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def curl_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.qf.get_quadrature_points_and_weights()
        cphi = self.curl_basis(bcs) #(NQ, NC, ldof, GD)
        A = bm.einsum("cqlg, cqdg, c, q->cld", cphi, cphi, cm, ws) #(NC, ldof, ldof)

        I = bm.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = bm.broadcast_to(c2d[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return B

    # def projection(self, f, method="L2"):
    #     M = self.mass_matrix()
    #     b = self.source_vector(f)
    #     x, _ = cg(M, b, tol="1e-12")
    #     return self.function(array=x)

    # def function(self, dim=None, array=None, dtype=bm.float64):
    #     if array is None:
    #         gdof = self.dof.number_of_global_dofs()
    #         array = bm.zeros(gdof, dtype=self.ftype)
    #     return Function(self, dim=dim, array=array, coordtype='barycentric', dtype=dtype)

    # def interplation(self, f):
    #     dv = self.dof_vector() #(NC, ldof, 3)
    #     ldof = self.dof.number_of_local_dofs()
    #     point = self.lspace.interpolation_points() #(NC, ldof//3, 3)
    #     lcell2dof = self.lspace.dof.cell_to_dof()
    #     cell2dof = self.dof.cell_to_dof()
    #     fval = f(point)[lcell2dof]

    #     fh = self.function()
    #     fh[cell2dof[:, :ldof//3]] = bm.sum(fval*dv[:, :ldof//3], axis=-1)
    #     fh[cell2dof[:, ldof//3:2*ldof//3]] = bm.sum(fval*dv[:, ldof//3:2*ldof//3], axis=-1)
    #     fh[cell2dof[:, 2*ldof//3:]] = bm.sum(fval*dv[:, 2*ldof//3:], axis=-1)
    #     return fh

    def set_dirichlet_bc(self, gd, uh, threshold=None, q=None,method=None):
        p = self.p
        mesh = self.mesh
        ldof = (p+1)*(p+2) 
        gdof = self.number_of_global_dofs()
       
        if type(threshold) is bm.array:
            index = threshold
        else:
            index = self.mesh.boundary_face_index()

        face2dof = self.dof.face_to_dof()[index]
        f2v = self.face_dof_vector(index=index) #(NF, ldof, 3)

        #bcs = self.lspace.multi_index_matrix[2](p)/p
        bcs = self.mesh.multi_index_matrix(p, 2,dtype=self.ftype)/p
        point = mesh.bc_to_point(bcs, index=index) #.swapaxes(0, 1) #(ldof//2, NF, 3)
        # nor = mesh.face_unit_normal()[index] #(NF, 3)

        # gval = gD(point, nor[:, None]) #(NF, ldof//2, 3)
        # gval = bm.cross(gval, nor[:, None])
        # print(bm.max(bm.abs(gval)))
        n = mesh.face_unit_normal()[index]
        n = n[:,None,:]
        gval = gd(point,n)
        gval = bm.cross(n,gval)


        uh[face2dof[:, :ldof//2]] = bm.sum(gval*f2v[:, :ldof//2], axis=-1)
        uh[face2dof[:, ldof//2:]] = bm.sum(gval*f2v[:, ldof//2:], axis=-1)
        # uh[face2dof[:, :ldof//2]] =0
        # uh[face2dof[:, ldof//2:]] =0
        isDDof = bm.zeros(gdof, device=self.device,dtype=bm.bool)
        isDDof[face2dof] = True
        return uh,isDDof
    
    boundary_interpolate = set_dirichlet_bc

    def set_neumann_bc(self,gD):
        p = self.p
        mesh = self.mesh
        isbdFace = mesh.boundary_face_flag()
        face2dof = self.dof.face_to_dof()[isbdFace]
        fm = mesh.entity_measure('face')[isbdFace]
        gdof = self.dof.number_of_global_dofs()
       
        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = self.face_basis(bcs)[isbdFace]
        points = mesh.bc_to_point(bcs)[isbdFace]
        n = mesh.face_unit_normal()[isbdFace]
        hval = gD(points,n)
        vec = bm.zeros(gdof, device=self.device,dtype=self.ftype)
        k = bm.einsum('fqg, fqlg,q,f->fl', hval, bphi,ws,fm) # (NF, ldof)
        bm.add.at(vec,face2dof,k)
        #bm.scatter_add(vec,face2dof,k)
        
        return -vec
    
    def cell_to_dof(self):
        return self.dof.cell2dof
    
    def face_to_dof(self):
        return self.dof.face_to_dof()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)
    
    def is_boundary_dof(self, threshold=None,method=None):
        return self.dof.is_boundary_dof()

    @barycentric
    def value(self, uh, bc, index=_S):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        #val = bm.einsum("cl, ...clk->...ck", uh[c2d], phi)
        val = bm.einsum("cl, cqlk->cqk", uh[c2d], phi)
        return val

    @barycentric
    def curl_value(self, uh, bc, index=_S):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.curl_basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        # val = bm.einsum("cl, ...clk->...ck", uh[c2d], phi)
        val = bm.einsum("cl, cqlg->cqg", uh[c2d], phi)
        return val

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
        errval = bm.sum((uval-uhval)*(uval-uhval), axis=-1)#(NQ, NC)
        val = bm.einsum("qc, q, c->", errval, ws, cm)
        return bm.sqrt(val)

    def curl_error(self, curlu, uh):
        '''@
        @brief 计算 ||u - uh||_{L_2}
        '''
        mesh = self.mesh
        cm = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        cuval = curlu(p) #(NQ, NC, GD)
        cuhval = uh.curl_value(bcs) #(NQ, NC, GD)
        errcval = bm.sum((cuval-cuhval)**2, axis=-1)#(NQ, NC)
        val = bm.einsum("qc, q, c->", errcval, ws, cm)
        return bm.sqrt(val)

    def source_vector(self, f):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        if hasattr(f, 'coordtype'):
            if f.coordtype == 'cartesian':
                pp = self.mesh.bc_to_point(bcs)
                fval = f(pp)
            elif f.coordtype == 'barycentric':
                fval = f(bcs)
        else:
            pp = mesh.bc_to_point(bcs) #(NQ, NC, GD)
            fval = f(pp) #(NQ, NC, GD)

        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        val = bm.einsum("cqg, cqlg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = bm.zeros(gdof, dtype=self.ftype)
        bm.add.at(vec, c2d, val)
        return vec
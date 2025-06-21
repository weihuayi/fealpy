
from typing import Union, TypeVar, Generic, Callable,Optional

from ..backend import TensorLike
from ..backend import backend_manager as bm
from .space import FunctionSpace
from . import BernsteinFESpace  


from scipy.sparse import csr_matrix
from ..mesh.mesh_base import Mesh
from ..decorator import barycentric, cartesian
import itertools

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class FirstNedelecDof3d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = mesh.multi_index_matrix(p,3)
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return 6*(p+1)+4*(p+1)*p+(p+1)*(p-1)*p//2
        elif doftype in {'cell', 3}: # number of dofs on each edge 
            return (p+1)*(p-1)*p//2
        elif doftype in {'face', 2}: # number of dofs inside the cell 
            return (p+1)*p
        elif doftype in {'edge', 1}: # number of dofs on each edge 
            return p+1
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        fdof = self.number_of_local_dofs(doftype='face')
        cdof = self.number_of_local_dofs(doftype='cell')
        return NE*edof + NC*cdof + NF*fdof

    def edge_to_dof(self, index=_S):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        return bm.arange(NE*edof, device=self.device).reshape(NE, edof)[index]

    def face_to_internal_dof(self, index=_S):
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        edof = self.number_of_local_dofs(doftype='edge')
        fdof = self.number_of_local_dofs(doftype='face')
        return bm.arange(NE*edof, NE*edof+NF*fdof,device=self.device).reshape(NF, fdof)[index]

    def face_to_dof(self,index=_S):
        p = self.p
        fldof = self.number_of_local_dofs('face')
        eldof = self.number_of_local_dofs('edge')
        fdof = fldof + eldof*3
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()

        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        f2e = self.mesh.face_to_edge()
        edge = self.mesh.entity('edge')
        face = self.mesh.entity('face')

        f2d = bm.zeros((NF, fdof), device=self.device,dtype=self.itype)
        #f2d[:, :eldof*3] = e2dof[f2e].reshape(NF, eldof*3)
        f2d = bm.set_at(f2d,(slice(None),slice(eldof*3)),e2dof[f2e].reshape(NF, eldof*3))
        s = [1, 0, 0]
        for i in range(3):
            flag = face[:, s[i]] == edge[f2e[:, i], 1]
            #f2d[flag, eldof*i:eldof*(i+1)] = f2d[flag, eldof*i:eldof*(i+1)][:, ::-1]
            k = bm.flip(f2d[flag, eldof*i:eldof*(i+1)],axis=-1)
            f2d = bm.set_at(f2d,(flag,slice(eldof*i,eldof*(i+1))),k)
            #f2d = bm.set_at(f2d,(flag,slice(eldof*i,eldof*(i+1))),f2d[flag, eldof*i:eldof*(i+1)][:, ::-1])
        #f2d[:, eldof*3:] = self.face_to_internal_dof() 
        f2d = bm.set_at(f2d,(slice(None),slice(eldof*3,eldof*3+fldof)),self.face_to_internal_dof())
        return f2d[index]

    def cell_to_dof(self):
        p = self.p
        cldof = self.number_of_local_dofs('cell')
        fldof = self.number_of_local_dofs('face')
        fldof_2 = fldof//2
        eldof = self.number_of_local_dofs('edge')
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()
        f2dof = self.face_to_internal_dof()

        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        c2e = self.mesh.cell_to_edge()
        c2f = self.mesh.cell_to_face()
        edge = self.mesh.entity('edge')
        face = self.mesh.entity('face')
        cell = self.mesh.entity('cell')

        c2d = bm.zeros((NC, ldof), dtype=bm.int64, device=self.device)
        # c2d[:, :eldof*6] = e2dof[c2e].reshape(NC, eldof*6)
        c2d = bm.set_at(c2d,(slice(None),slice(eldof*6)),e2dof[c2e].reshape(NC, eldof*6))
        s = [0, 0, 0, 1, 1, 2]
        for i in range(6):
            flag = cell[:, s[i]] == edge[c2e[:, i], 1]
            # c2d[flag, eldof*i:eldof*(i+1)] = bm.flip(c2d[flag, eldof*i:eldof*(i+1)],axis=-1)
            c2d = bm.set_at(c2d,(flag,slice(eldof*i,eldof*(i+1))),bm.flip(c2d[flag, eldof*i:eldof*(i+1)],axis=-1))
        if fldof > 0:
            locFace = bm.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],device=self.device, dtype=self.itype)
            midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]
            
            midx = self.mesh.multi_index_matrix(p-1, 2)
            perms = bm.array(list(itertools.permutations([0, 1, 2])), device=self.device)
            indices = bm.zeros((6, len(midx)),device=self.device, dtype=self.itype)
            for i in range(6):
                indices[i] = midx2num(midx[:, perms[i]])

            c2fp = self.mesh.cell_to_face_permutation(locFace=locFace)

            perm2num = lambda a : a[:, 0]*2 + (a[:, 1]>a[:, 2]) 
            for i in range(4):
                #perm = np.argsort(c2fp[:, i], axis=1)
                perm =c2fp[:, i]
                pnum = perm2num(perm)
                N = eldof*6+fldof*i
                # c2d[:, N:N+fldof_2] = f2dof[c2f[:, i, None], indices[None, pnum]]
                # c2d[:, N+fldof_2:N+fldof] = f2dof[c2f[:, i, None], fldof_2 +indices[None, pnum]]
                c2d = bm.set_at(c2d,(slice(None),slice(N,N+fldof_2)),f2dof[c2f[:, i, None], indices[None, pnum]])
                c2d = bm.set_at(c2d,(slice(None),slice(N+fldof_2,N+fldof)),f2dof[c2f[:, i, None], fldof_2 +indices[None, pnum]])
        
        if cldof > 0:
            #c2d[:, eldof*6+fldof*4:] = bm.arange(NE*eldof+NF*fldof, gdof).reshape(NC, cldof) 
            c2d = bm.set_at(c2d,(slice(None),slice(eldof*6+fldof*4,eldof*6+fldof*4+cldof,None)),bm.arange(NE*eldof+NF*fldof, gdof).reshape(NC, cldof))
        return c2d

    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        edge = self.mesh.entity('edge')
        bdnflag = self.mesh.boundary_node_flag()
        bdeflag = bm.all(bdnflag[edge], axis=1)
        eidx = bm.nonzero(bdeflag)[0] 
        e2d = self.edge_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        p = self.p
        mesh = self.mesh
        ldof = p+1
        gdof = self.number_of_global_dofs()
       
        isDDof = bm.zeros(gdof, device=self.device,dtype=bm.bool)
        index = self.mesh.boundary_face_index()
        face2dof = self.face_to_internal_dof()[index]
        isDDof[face2dof] = True

        # edge = self.mesh.entity('edge')
        # bdnflag = self.mesh.boundary_node_flag()
        # bdeflag = bm.all(bdnflag[edge], axis=1)
        NE = mesh.number_of_edges()
        f2e = mesh.face_to_edge()[index]
        bdeflag = bm.zeros(NE,device=self.device,dtype=bm.bool)
        bdeflag[f2e] = True
        index = bm.nonzero(bdeflag)[0]

        edge2dof = self.edge_to_dof()[index]
        isDDof[edge2dof] = True

        return isDDof

class FirstNedelecFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.dof = FirstNedelecDof3d(mesh, p)

        self.bspace = BernsteinFESpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.qf = self.mesh.quadrature_formula(p+3)

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        #TODO:JAX
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def cross(self, a, b):
        if bm.backend_name == 'numpy':
            return bm.cross(a, b)
        elif bm.backend_name == 'pytorch':
            return bm.linalg.cross(a, b)

    @barycentric
    def basis(self, bcs, index=_S):

        import time
        t = time.time()
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        cldof = self.dof.number_of_local_dofs("cell")
        fldof = self.dof.number_of_local_dofs("face")
        eldof = self.dof.number_of_local_dofs("edge")
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda()
        ledge = mesh.localEdge

        c2esign = mesh.cell_to_edge_sign() #(NC, 3, 2)

        l = bm.zeros((4, )+bcs[None,: ,0,None, None].shape,device=self.device, dtype=self.ftype)

        l = bm.set_at(l,(0),bcs[None, :,0,None, None])
        l = bm.set_at(l,(1),bcs[None, :,1,None, None])
        l = bm.set_at(l,(2),bcs[None, :,2,None, None])
        l = bm.set_at(l,(3),bcs[None, :,3,None, None])
        #l = np.tile(l, (1, NC, 1, 1))

        # edge basis
        phi = self.bspace.basis(bcs, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        val = bm.zeros((NC,) + bcs.shape[:-1]+(ldof, 3),device=self.device, dtype=self.ftype)
        locEdgeDual = bm.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]])
        for i in range(6):
            flag = bm.all(multiIndex[:, locEdgeDual[i]]==0, axis=1)
            phie = phi[:, :, flag] 
            c2esi = c2esign[:, i] 
            v = l[ledge[i, 0]]*glambda[:,None, ledge[i, 1], None,:] - l[ledge[i, 1]]*glambda[:,None, ledge[i, 0], None,:]
            v = bm.set_at(v,(~c2esi,slice(None),slice(None),slice(None)),-v[~c2esi, :, :, :])
            val = bm.set_at(val,(...,slice(eldof*i,eldof*(i+1)),slice(None)),phie[..., None]*v)
        # face basis
        if(p > 0):
            phi = self.bspace.basis(bcs, p=p-1)
            multiIndex = self.mesh.multi_index_matrix(p-1, 3)
            permcf = self.mesh.cell_to_face_permutation()
            #permcf = bm.argsort(permcf, axis=-1)
            localFace = self.mesh.localFace
            for i in range(4):

                flag = multiIndex[:, i]==0
                phif = phi[:, :, flag] 

                permci = localFace[i, permcf[:, i]] #(NC, 3)
                #l0 = l[permci[:, 0], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l1 = l[permci[:, 1], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l2 = l[permci[:, 2], ..., np.arange(NC), :, :].swapaxes(0, 1)
                l0 = l[permci[:, 0], 0,:, :, :]
                l1 = l[permci[:, 1], 0,:, :, :]
                l2 = l[permci[:, 2], 0,:, :, :]

                g0 = glambda[bm.arange(NC),None, permci[:, 0], None]
                g1 = glambda[bm.arange(NC),None, permci[:, 1], None]
                g2 = glambda[bm.arange(NC),None, permci[:, 2], None]

                v0 = l2*(l0*g1 - l1*g0)
                v1 = l0*(l1*g2 - l2*g1)

                N = eldof*6+fldof*i 
                val = bm.set_at(val,(...,slice(N,N+fldof//2),slice(None)),v0*phif[..., None])
                val = bm.set_at(val,(...,slice(N+fldof//2,N+fldof),slice(None)),v1*phif[..., None])

        if(p > 1):
            phi = self.bspace.basis(bcs, p=p-2) #(NQ, NC, cldof)
            v0 = l[2]*l[3]*(l[0]*glambda[:,None, 1, None,:] - l[1]*glambda[:,None, 0, None,:]) #(NQ, NC, ldof, 2)
            v1 = l[0]*l[3]*(l[1]*glambda[:,None, 2, None,:] - l[2]*glambda[:,None, 1, None,:]) #(NQ, NC, ldof, 2)
            v2 = l[0]*l[1]*(l[2]*glambda[:,None, 3, None,:] - l[3]*glambda[:,None, 2, None,:]) #(NQ, NC, ldof, 2)

            N = eldof*6+fldof*4
            val = bm.set_at(val,(...,slice(N,N+cldof//3),slice(None)),v0*phi[..., None])
            val = bm.set_at(val,(...,slice(N+cldof//3,N+2*cldof//3),slice(None)),v1*phi[..., None])
            val = bm.set_at(val,(...,slice(N+2*cldof//3,N+cldof),slice(None)),v2*phi[..., None])
        s = time.time()
        print("tt : ", s-t)

        return val
    
    def face_internal_basis(self, bcs, index=_S):
        p = self.p
        assert(p>0)

        mesh = self.mesh
        NF = mesh.number_of_faces()
        GD = mesh.geo_dimension()
        
        fdof = self.dof.number_of_local_dofs("face")
        glambda = mesh.grad_face_lambda()
        val = bm.zeros((NF,) + bcs.shape[:-1]+(fdof, 3),device=self.device, dtype=self.ftype)
        
        l = bm.zeros((3, )+bcs[None, :,0, None, None].shape, device=self.device,dtype=self.ftype)
        l = bm.set_at(l,(0),bcs[None, :,0,  None, None])
        l = bm.set_at(l,(1),bcs[None, :,1,  None, None])
        l = bm.set_at(l,(2),bcs[None, :,2,  None, None])

        phi = self.bspace.basis(bcs, p=p-1)
        v0 = l[2]*(l[0]*glambda[:,None, 1, None] - l[1]*glambda[:,None, 0, None]) 
        v1 = l[0]*(l[1]*glambda[:,None, 2, None] - l[2]*glambda[:,None, 1, None]) 
        # val[..., :fdof//2, :] = v0*phi[..., None]
        # val[..., fdof//2:, :] = v1*phi[..., None]
        val = bm.set_at(val,(...,slice(0,fdof//2),slice(None)),v0*phi[..., None])
        val = bm.set_at(val,(...,slice(fdof//2,fdof),slice(None)),v1*phi[..., None])
        return val

    @barycentric
    def face_basis(self, bcs, index=_S):
        p = self.p
        mesh = self.mesh
        NF = mesh.number_of_faces()
        GD = mesh.geo_dimension()
        fldof = self.dof.number_of_local_dofs("face")
        eldof = self.dof.number_of_local_dofs("edge")
        ldof = 3*eldof + fldof
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda(TD =2)
        ledge = bm.array([[1, 2],
                         [2, 0],
                         [0, 1]],device=self.device)
        c2esign = mesh.face_to_edge_sign()
        l = bm.zeros((3, )+bcs[None, :,0, None, None].shape,device=self.device, dtype=self.ftype)
        l = bm.set_at(l,(0),bcs[None, :,0,  None, None])
        l = bm.set_at(l,(1),bcs[None, :,1,  None, None])
        l = bm.set_at(l,(2),bcs[None, :,2,  None, None])
        # edge basis
        phi = self.bspace.basis(bcs, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 2)
        val = bm.zeros((NF,) + bcs.shape[:-1]+(ldof, 3),device=self.device, dtype=self.ftype)
        for i in range(3):
            phie = phi[:, :, multiIndex[:, i]==0]
            c2esi = c2esign[:, i]
            v = l[ledge[i, 0]]*glambda[:,None, ledge[i, 1], None,:] - l[ledge[i, 1]]*glambda[:,None, ledge[i, 0], None,:]
            v = bm.set_at(v,(~c2esi,slice(None),slice(None),slice(None)),-v[~c2esi, :,  :, :])
            val = bm.set_at(val,(...,slice(eldof*i,eldof*(i+1)),slice(None)),phie[..., None]*v)
        # cell basis
        if(p > 0):
            phi = self.bspace.basis(bcs, p=p-1)
            v0 = l[2]*(l[0]*glambda[:,None, 1, None] - l[1]*glambda[:,None, 0, None])
            v1 = l[0]*(l[1]*glambda[:,None, 2, None] - l[2]*glambda[:,None, 1, None])
            val = bm.set_at(val,(...,slice(eldof*3,eldof*3+fldof//2),slice(None)),v0*phi[..., None])
            val = bm.set_at(val,(...,slice(eldof*3+fldof//2,None),slice(None)),v1*phi[..., None] )
        return val[index]


    @barycentric
    def curl_basis(self, bcs):

        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        cldof = self.dof.number_of_local_dofs("cell")
        fldof = self.dof.number_of_local_dofs("face")
        eldof = self.dof.number_of_local_dofs("edge")
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda()
        ledge = mesh.localEdge

        c2esign = mesh.cell_to_edge_sign() #(NC, 6, 2)

        l = bm.zeros((4, )+bcs[None,:, 0,None, None].shape,device=self.device, dtype=self.ftype)

        l = bm.set_at(l,(0),bcs[None, :,0,None, None])
        l = bm.set_at(l,(1),bcs[None, :,1,None, None])
        l = bm.set_at(l,(2),bcs[None, :,2,None, None])
        l = bm.set_at(l,(3),bcs[None, :,3,None, None])

        #l = np.tile(l, (1, NC, 1, 1))

        phi = self.bspace.basis(bcs, p=p)
        gphi = self.bspace.grad_basis(bcs, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        val = bm.zeros((NC,)+bcs.shape[:-1]+(ldof, 3),device=self.device, dtype=self.ftype)

        # edge basis
        locEdgeDual = bm.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]])
        for i in range(6):
            flag = bm.all(multiIndex[:, locEdgeDual[i]]==0, axis=1)
            phie = phi[..., flag]
            gphie = gphi[..., flag, :]
            c2esi = c2esign[:, i]
            v = l[ledge[i, 0]]*glambda[:,None, ledge[i, 1], None,:] - l[ledge[i, 1]]*glambda[:,None, ledge[i, 0], None,:]
            v = bm.set_at(v,(~c2esi,slice(None),slice(None),slice(None)),-v[~c2esi, :, :, :])
            cv = 2*self.cross(glambda[:,None, ledge[i, 0],None,:], glambda[:,None, ledge[i, 1],None,:]) #(NC, )
            cv = bm.set_at(cv,(~c2esi),-cv[~c2esi])
            val = bm.set_at(val,(...,slice(eldof*i,eldof*(i+1)),slice(None)),phie[..., None]*cv + self.cross(gphie, v))
        
        # face basis
        if(p > 0):
            phi = self.bspace.basis(bcs, p=p-1) #(NQ, NC, cldof)
            gphi = self.bspace.grad_basis(bcs, p=p-1)
            multiIndex = self.mesh.multi_index_matrix(p-1, 3)
            permcf = self.mesh.cell_to_face_permutation()
            localFace = self.mesh.localFace
            for i in range(4):

                flag = multiIndex[:, i]==0
                phif = phi[..., flag]
                gphif = gphi[..., flag, :]

                permci = localFace[i, permcf[:, i]] #(NC, 3)
                #l0 = l[permci[:, 0], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l1 = l[permci[:, 1], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l2 = l[permci[:, 2], ..., np.arange(NC), :, :].swapaxes(0, 1)

                l0 = l[permci[:, 0], 0, :, :, :]
                l1 = l[permci[:, 1], 0, :, :, :]
                l2 = l[permci[:, 2], 0, :, :, :]

                g0 = glambda[bm.arange(NC),None, permci[:, 0], None]
                g1 = glambda[bm.arange(NC),None, permci[:, 1], None]
                g2 = glambda[bm.arange(NC),None, permci[:, 2], None]

                v0 = l2*(l0*g1 - l1*g0) #(NQ, NC, fldof//2, 2)
                v1 = l0*(l1*g2 - l2*g1)

                cv0 = self.cross(g2, (l0*g1 - l1*g0)) + 2*l2*self.cross(g0, g1)
                cv1 = self.cross(g0, (l1*g2 - l2*g1)) + 2*l0*self.cross(g1, g2)

                N = eldof*6+fldof*i
                val = bm.set_at(val,(...,slice(N,N+fldof//2),slice(None)),-self.cross(v0, gphif)+phif[..., None]*cv0)
                val = bm.set_at(val,(...,slice(N+fldof//2,N+fldof),slice(None)),-self.cross(v1, gphif)+phif[..., None]*cv1)
        # cell basis
        if(p > 1):
            phi = self.bspace.basis(bcs, p=p-2)
            gphi = self.bspace.grad_basis(bcs, p=p-2)

            g0 = glambda[:,None, 0, None,:]
            g1 = glambda[:,None, 1, None,:]
            g2 = glambda[:,None, 2, None,:]
            g3 = glambda[:,None, 3, None,:]

            v0 = l[2]*l[3]*(l[0]*g1 - l[1]*g0)
            v1 = l[0]*l[3]*(l[1]*g2 - l[2]*g1)
            v2 = l[0]*l[1]*(l[2]*g3 - l[3]*g2)
            cv0 = self.cross(l[2]*g3+l[3]*g2, l[0]*g1-l[1]*g0) + 2*l[2]*l[3]*self.cross(g0, g1)
            cv1 = self.cross(l[0]*g3+l[3]*g0, l[1]*g2-l[2]*g1) + 2*l[0]*l[3]*self.cross(g1, g2)
            cv2 = self.cross(l[0]*g1+l[1]*g0, l[2]*g3-l[3]*g2) + 2*l[0]*l[1]*self.cross(g2, g3)

            N = eldof*6+fldof*4
            # val[..., N:N+cldof//3, :] = self.cross(gphi, v0) + phi[..., None]*cv0 
            # val[..., N+cldof//3:N+2*cldof//3, :] = self.cross(gphi, v1) + phi[...,None]*cv1  
            # val[..., N+2*cldof//3:N+cldof, :] = self.cross(gphi, v2) + phi[...,None]*cv2  
            val = bm.set_at(val,(...,slice(N,N+cldof//3),slice(None)),self.cross(gphi, v0)+phi[..., None]*cv0)
            val = bm.set_at(val,(...,slice(N+cldof//3,N+2*cldof//3),slice(None)),self.cross(gphi, v1)+phi[..., None]*cv1)
            val = bm.set_at(val,(...,slice(N+2*cldof//3,N+cldof),slice(None)),self.cross(gphi, v2)+phi[..., None]*cv2)
        return val

    def is_boundary_dof(self, threshold=None,method=None):
        return self.dof.is_boundary_dof()

    def cell_to_dof(self):
        return self.dof.cell2dof
           
    def face_to_dof(self,index=_S):
        return self.dof.face_to_dof()[index]

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bcs, index=_S):
        '''@
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.basis(bcs)
        c2d = self.dof.cell_to_dof()
        val = bm.einsum("cl, cqlk->cqk", uh[c2d], phi)
        return val

    @barycentric
    def curl_value(self, uh, bcs, index=_S):
        cphi = self.curl_basis(bcs)
        c2d = self.dof.cell_to_dof()
        val = bm.einsum("cl, cqli->cqi", uh[c2d], cphi)
        return val

    @barycentric
    def div_value(self, uh, bc, index=_S):
        pass

    @barycentric
    def grad_value(self, uh, bc, index=_S):
        pass

    @barycentric
    def edge_value(self, uh, bc, index=_S):
        pass

    @barycentric
    def face_value(self, uh, bcs, index=_S):
        phi = self.face_basis(bcs)[index]
        f2d = self.dof.face_to_dof()[index]
        val = bm.einsum("cl, cqlk->cqk", uh[f2d], phi)
        return val

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

        I = bm.broadcast_to(c2d[:, :, None], mass.shape).reshape(-1)
        J = bm.broadcast_to(c2d[:, None, :], mass.shape).reshape(-1)
        I = bm.to_numpy(I)
        J = bm.to_numpy(J)
        mass = bm.to_numpy(mass).reshape(-1)
        M = csr_matrix((mass, (I, J)), shape=(gdof, gdof))
        return M 

    def curl_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure

        c2d = self.dof.cell_to_dof()
        bcs, ws = self.qf.get_quadrature_points_and_weights()

        cphi = self.curl_basis(bcs) 
        A = bm.einsum("cqlg, cqdg, c, q->cld", cphi, cphi, cm, ws)

        I = bm.broadcast_to(c2d[:, :, None], A.shape).reshape(-1)
        J = bm.broadcast_to(c2d[:, None, :], A.shape).reshape(-1)
        I = bm.to_numpy(I)
        J = bm.to_numpy(J)
        A = bm.to_numpy(A).reshape(-1)
        B = csr_matrix((A, (I, J)), shape=(gdof, gdof))
        return B
    
    def face_mass_matrix(self, c = 1, index=_S):
        """
        @brief (n \times u, n \times v)_{Gamma_{robin}}
        @param c 系数, 现在只考虑了 c 是常数的情况
        """
        p = self.p
        mesh = self.mesh
        qf = mesh.quadrature_formula(p+2,"face")
        bcs, ws = qf.get_quadrature_points_and_weights()
        face2dof = self.dof.face_to_dof()[index]
        fm = self.mesh.entity_measure("face", index=index)
        fphi = self.face_basis(bcs, index=index)

        EMc = bm.einsum('fqlg, fqmg, q, f->flm', fphi, fphi, ws, fm)

        gdof = self.dof.number_of_global_dofs()
        I = bm.broadcast_to(face2dof[:,:, None], EMc.shape)
        J = bm.broadcast_to(face2dof[:, None, :], EMc.shape)
        EM = c*csr_matrix((EMc.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return EM

    def source_vector(self, f):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof()

        p = mesh.bc_to_point(bcs) 
        fval = f(p) 

        phi = self.basis(bcs) 
        val = bm.einsum("cqg, cqlg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = bm.zeros(gdof, dtype=self.ftype)
        bm.scatter_add(vec, c2d.reshape(-1), val.reshape(-1))
        return vec
    
    # def projection(self, f, method="L2"):
    #     M = self.mass_matrix()
    #     b = self.source_vector(f)
    #     x = spsolve(M, b)
    #     return self.function(array=x)

    # def interplation(self, f):
    #     mesh = self.mesh
    #     node = mesh.entity("node")
    #     edge = mesh.entity("edge")

    #     gdof = self.dof.number_of_global_dofs()
    #     e2n = mesh.edge_unit_normal()
    #     val = np.zeros(gdof, dtype=np.float_)

    #     f0 = f(node[edge[:, 0]]) 
    #     f1 = f(node[edge[:, 1]])

    #     val[0::2] = np.sum(f0*e2n, axis=1)
    #     val[1::2] = np.sum(f1*e2n, axis=1)
    #     return self.function(array=val)

    # dirichlet边界条件
    def set_dirichlet_bc(self, gd, uh, threshold=None, q=None,method=None):
        p = self.p
        mesh = self.mesh
        gdof = self.number_of_global_dofs()       
        isDDof = bm.zeros(gdof, device=self.device, dtype=bm.bool)

    
        index1 = self.mesh.boundary_face_index()
        # 边界面的点
        if p>0:
            face2dof = self.dof.face_to_dof()[index1]

            qf = mesh.quadrature_formula(p+2,"face")
            bcs, ws = qf.get_quadrature_points_and_weights()

            fbasis = self.face_basis(bcs)[index1] # (NF,NQ,ldof,GD)
            fm = mesh.entity_measure('face')[index1]
            M = bm.einsum("cqlg, cqmg, q, c->clm", fbasis, fbasis, ws, fm)
            Minv = bm.linalg.inv(M)

            points = mesh.bc_to_point(bcs)[index1]
            n = mesh.face_unit_normal()[index1]
            n = n[:,None,:]
            h2 = gd(points,n)

            #g = bm.cross(n, h2) 
            #g = bm.cross(g,n)
            g = bm.cross(n,h2)

            g1 = bm.einsum("cqld, cqd,q,c->cl", fbasis, g, ws, fm)

            uh[face2dof] = bm.einsum("cl, clm->cm", g1, Minv)
            isDDof[face2dof] = True

        # 边界边界的点
        NE = mesh.number_of_edges()
        f2e = mesh.face_to_edge()[index1]
        bdeflag = bm.zeros(NE, device=self.device, dtype=bm.bool)
        bdeflag[f2e] = True
        index2 = bm.nonzero(bdeflag)[0]
        edge2dof = self.dof.edge_to_dof()[index2]
        em = mesh.entity_measure('edge')[index2]
        t = mesh.edge_tangent()[index2]/em[:, None]
        
        # 右端矩阵组装
        qf = mesh.quadrature_formula(p+2,"edge")
        bcs, ws = qf.get_quadrature_points_and_weights()        
        bphi = self.bspace.basis(bcs, p=p)
        M = bm.einsum("eql, eqm, q->lm", bphi, bphi, ws)
        Minv = bm.linalg.inv(M)
        Minv = Minv*em[:,None,None]
        
        points1 = mesh.bc_to_point(bcs)[index2]

        n1 = mesh.face_unit_normal()
        n2 = bm.zeros((NE,3))
        f2e = mesh.face_to_edge()
        n2[f2e] = n1[:,None,:]
        n2 = n2[:,None,:][index2]

        h1 = gd(points1,n2)
        h1 = bm.cross(n2,h1)
        b = bm.einsum('eqd, ed->eq', h1, t) 
        
        g2 = bm.einsum('eql, eq,q->el', bphi, b,ws)

        uh[edge2dof] = bm.einsum('el, elm->em', g2, Minv)
        isDDof[edge2dof] = True
        return uh,isDDof


    boundary_interpolate = set_dirichlet_bc

    # Neumann边界条件
    def set_neumann_bc(self,gD):
        p = self.p
        mesh = self.mesh
        isbdFace = mesh.boundary_face_flag()
        face2dof = self.dof.face_to_dof()[isbdFace]
        fm = mesh.entity_measure('face')[isbdFace]
        gdof = self.dof.number_of_global_dofs()

        # Bernstein 空间的单位质量矩阵
        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = self.face_basis(bcs)[isbdFace]
        points = mesh.bc_to_point(bcs)[isbdFace]
        n = mesh.face_unit_normal()[isbdFace]
        hval = gD(points,n)
        vec = bm.zeros(gdof, dtype=self.ftype)
        k = bm.einsum('fqg, fqlg,q,f->fl', hval, bphi,ws,fm) # (NF, ldof)
        bm.add.at(vec,face2dof,k)
        #bm.scatter_add(vec,face2dof,k)
        
        return -vec
    









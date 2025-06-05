
from typing import Union, TypeVar, Generic, Callable,Optional
import itertools

from ..backend import TensorLike
from ..backend import backend_manager as bm
from .space import FunctionSpace
from .bernstein_fe_space import BernsteinFESpace
from .lagrange_fe_space import LagrangeFESpace  
from .function import Function

from scipy.sparse import csr_matrix
from ..mesh.mesh_base import Mesh
from ..decorator import barycentric, cartesian

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)


class RTDof3d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.ftype = mesh.ftype 
        self.itype = mesh.itype
        self.device = mesh.device

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+2)*(p+4)//2
        elif doftype in {'cell', 3}: # number of dofs on each edge 
            return (p+1)*(p+2)*p//2
        elif doftype in {'face', 2}: # number of dofs inside the cell 
            return (p+1)*(p+2)//2
        elif doftype in {'edge', 1}: # number of dofs on each edge 
            return 0
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face')
        cdof = self.number_of_local_dofs(doftype='cell')
        return NC*cdof + NF*fdof

    def face_to_dof(self, index=_S):
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face')
        return bm.arange(NF*fdof,device=self.device).reshape(NF, fdof)[index]

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cldof = self.number_of_local_dofs('cell')
        fldof = self.number_of_local_dofs('face')
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        f2dof = self.face_to_dof()

        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        c2f = self.mesh.cell_to_face()
        face = self.mesh.entity('face')
        cell = self.mesh.entity('cell')

        c2d = bm.zeros((NC, ldof),device=self.device, dtype=self.itype)

        locFace = bm.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],device=self.device, dtype=self.itype)
        midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]

        midx = mesh.multi_index_matrix(p, 2)
        perms = bm.array(list(itertools.permutations([0, 1, 2])))
        indices = bm.zeros((6, len(midx)),device=self.device, dtype=self.itype)
        for i in range(6):
            indices[i] = midx2num(midx[:, perms[i]])

        c2fp = self.mesh.cell_to_face_permutation(locFace=locFace)

        perm2num = lambda a : a[:, 0]*2 + (a[:, 1]>a[:, 2]) 
        for i in range(4):
            perm =c2fp[:, i]
            pnum = perm2num(perm)
            N = fldof*i
            #c2d[:, N:N+fldof] = f2dof[c2f[:, i, None], indices[None, pnum]]
            c2d = bm.set_at(c2d,(slice(None), slice(N, N+fldof)),f2dof[c2f[:, i, None], indices[None, pnum]]) 
        if cldof > 0:
            #c2d[:, fldof*4:] = bm.arange(NF*fldof, gdof,device=self.device).reshape(NC, cldof) 
            c2d = bm.set_at(c2d,(slice(None), slice(fldof*4, None)),bm.arange(NF*fldof, gdof,device=self.device).reshape(NC, cldof)) 
        return c2d

    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        eidx = self.mesh.boundary_face_index()
        e2d = self.face_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        bddof = self.boundary_dof()

        gdof = self.number_of_global_dofs()
        flag = bm.zeros(gdof, device=self.device, dtype=bm.bool)

        flag[bddof] = True
        return flag


class RaviartThomasFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.dof = RTDof3d(mesh, p)

        self.bspace = BernsteinFESpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.qf = self.mesh.quadrature_formula(p+3)
        self.ftype = mesh.ftype
        self.itype = mesh.itype

        #TODO:JAX
        self.device = mesh.device

    @barycentric
    def basis(self, bcs,index=_S):

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

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        l = bm.zeros((4, )+bcs[None,:, 0, None, None].shape,device=self.device, dtype=self.ftype)
        l[0] = bcs[None, :, 0, None, None]
        l[1] = bcs[None, :, 1, None, None]
        l[2] = bcs[None, :, 2, None, None]
        l[3] = bcs[None, :, 3, None, None] #(NC, NQ, ldof, 2)

        # face basis
        val = bm.zeros((NC,)+bcs.shape[:-1]+(ldof, 3), device=self.device, dtype=self.ftype)
        phi = self.bspace.basis(bcs, p=p) #(NC, NQ, cldof)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        cm = self.mesh.entity_measure("cell")
        c2fs = self.mesh.cell_to_face_sign()
        lf = self.mesh.localFace
        for i in range(4):
            c2fsi = c2fs[:, i]
            flag = multiIndex[:, i]==0
            phif = phi[:, :, flag] #(NC, NQ, fldof//2)

            l0 = l[lf[i, 0]]
            l1 = l[lf[i, 1]]
            l2 = l[lf[i, 2]]

            v0 = (node[cell[:, lf[i, 0]]] - node[cell[:, i]])/cm[:, None]
            v1 = (node[cell[:, lf[i, 1]]] - node[cell[:, i]])/cm[:, None]
            v2 = (node[cell[:, lf[i, 2]]] - node[cell[:, i]])/cm[:, None]

            v = l0*v0[:, None,None,:] + l1*v1[:,None,None,:] + l2*v2[:,None, None,:] #(NQ, NC, fldof, 3)
            #v[~c2fsi,:, :, :] *= -1
            v = bm.set_at(v, (~c2fsi, slice(None), slice(None), slice(None)), -v[~c2fsi,:, :, :])

            N = fldof*i
            #val[..., N:N+fldof, :] = v*phif[..., None]
            val = bm.set_at(val,(..., slice(None), slice(N, N+fldof), slice(None)),v*phif[..., None])

        if(p > 0):
            phi = self.bspace.basis(bcs, p=p-1) #(NC, NQ, cldof)
            for i in range(3):
                l0 = l[lf[i, 0]]
                l1 = l[lf[i, 1]]
                l2 = l[lf[i, 2]]

                v0 = (node[cell[:, lf[i, 0]]] - node[cell[:, i]])/cm[:, None]
                v1 = (node[cell[:, lf[i, 1]]] - node[cell[:, i]])/cm[:, None]
                v2 = (node[cell[:, lf[i, 2]]] - node[cell[:, i]])/cm[:, None]

                v = l[i]*(l0*v0[:, None,None,:] + l1*v1[:, None,None,:] + l2*v2[:, None,None,:]) #(NQ, NC, fldof, 3)

                N = fldof*4+i*cldof//3
                #val[..., N:N+cldof//3, :] = v*phi[..., None]
                val = bm.set_at(val,(..., slice(None), slice(N, N+cldof//3), slice(None)),v*phi[..., None])
        return val[index]

    @barycentric
    def div_basis(self, bcs,index=_S):

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

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        l = bm.zeros((4, )+bcs[None, :,0,None, None].shape,device=self.device, dtype=self.ftype)
        l[0] = bcs[None, :, 0, None, None]
        l[1] = bcs[None, :, 1, None, None]
        l[2] = bcs[None, :, 2, None, None]
        l[3] = bcs[None, :, 3, None, None] #(NC, NQ, ldof, 1)

#         l = np.tile(l, (1, NC, 1, 1))

        phi = self.bspace.basis(bcs, p=p)
        gphi = self.bspace.grad_basis(bcs, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        val = bm.zeros((NC,)+bcs.shape[:-1]+(ldof,), device=self.device, dtype=self.ftype)

        # face basis
        phi = self.bspace.basis(bcs, p=p) #(NQ, NC, cldof)
        gphi = self.bspace.grad_basis(bcs, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        cm = self.mesh.entity_measure("cell")
        c2fs = self.mesh.cell_to_face_sign()
        lf = self.mesh.localFace
        for i in range(4):
            c2fsi = c2fs[:, i]
            flag = multiIndex[:, i]==0
            phif = phi[:, :, flag] #(NQ, NC, fldof//2)
            gphif = gphi[:, :, flag] #(NQ, NC, fldof//2)

            l0 = l[lf[i, 0]]
            l1 = l[lf[i, 1]]
            l2 = l[lf[i, 2]]

            g0 = glambda[:, lf[i, 0], None] #(NC, 1, 3)
            g1 = glambda[:, lf[i, 1], None]
            g2 = glambda[:, lf[i, 2], None]

            v0 = (node[cell[:, lf[i, 0]]] - node[cell[:, i]])/cm[:, None] #(NC, 1, 3)
            v1 = (node[cell[:, lf[i, 1]]] - node[cell[:, i]])/cm[:, None]
            v2 = (node[cell[:, lf[i, 2]]] - node[cell[:, i]])/cm[:, None]

            v = l0*v0[:, None,None,:] + l1*v1[:, None,None,:] + l2*v2[:, None,None,:] #(NQ, NC, fldof, 3)
            #v[~c2fsi,:, :, :] *= -1
            v = bm.set_at(v, (~c2fsi, slice(None), slice(None), slice(None)), -v[~c2fsi,:, :, :])
            dv = bm.sum(g0*v0[:, None], axis=-1) + bm.sum(g1*v1[:, None], axis=-1) + bm.sum(g2*v2[:, None], axis=-1)
            dv[~c2fsi] *= -1

            N = fldof*i
            #val[..., N:N+fldof] = dv[:,None,:]*phif+bm.sum(gphif*v, axis=-1)
            val = bm.set_at(val,(...,slice(N, N+fldof)),dv[:,None,:]*phif+bm.sum(gphif*v, axis=-1))

        if(p > 0):
            phi = self.bspace.basis(bcs, p=p-1) #(NQ, NC, cldof)
            gphi = self.bspace.grad_basis(bcs, p=p-1) #(NQ, NC, cldof)
            for i in range(3):
                l0 = l[lf[i, 0]]
                l1 = l[lf[i, 1]]
                l2 = l[lf[i, 2]]

                g0 = glambda[:, lf[i, 0], None]
                g1 = glambda[:, lf[i, 1], None]
                g2 = glambda[:, lf[i, 2], None]
                gi = glambda[:, i, None]

                v0 = (node[cell[:, lf[i, 0]]] - node[cell[:, i]])/cm[:, None]
                v1 = (node[cell[:, lf[i, 1]]] - node[cell[:, i]])/cm[:, None]
                v2 = (node[cell[:, lf[i, 2]]] - node[cell[:, i]])/cm[:, None]

                w = (l0*v0[:, None,None,:] + l1*v1[:, None,None,:] + l2*v2[:, None,None,:]) #(NQ, NC, fldof, 3)
                dw = bm.sum(g0*v0[:, None], axis=-1) + bm.sum(g1*v1[:, None], axis=-1) + bm.sum(g2*v2[:, None], axis=-1)
                v = l[i]*w #(NQ, NC, fldof, 3)
                dv = bm.sum(gi[:,None]*w, axis=-1) + l[i, ..., 0]*dw[:,None] 

                N = fldof*4+i*cldof//3
                #val[..., N:N+cldof//3] = dv*phi + bm.sum(gphi*v, axis=-1)
                val = bm.set_at(val,(...,slice(N, N+cldof//3)),dv*phi + bm.sum(gphi*v, axis=-1))
        return val[index]
    
    @barycentric
    def face_basis(self, bcs, index=_S):
        # fn = self.mesh.face_normal(index=index)
        # fn /= bm.sqrt(bm.sum(fn**2, axis=1)[:, None])

        # fphi = self.bspace.basis(bcs, p=self.p) #(NQ, NE, eldof)
        # val = fphi[..., None] * fn[:, None,None,:]
        fn = self.mesh.face_unit_normal(index=index)
        fm = self.mesh.entity_measure('face')[index]
        fm = 1/fm     
        fm = 3*fm
        fphi = self.bspace.basis(bcs, p=self.p) #(NC, NQ, eldof)
        val0 = fphi * fm[:, None,None]

        val = val0[:,:,:,None] * fn[:,None,None,:]

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
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        cphi = self.div_basis(bcs)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = bm.einsum("cl, cql->cq", uh[c2d], cphi)
        return val

    @barycentric
    def curl_value(self, uh, bc, index=_S):
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
    
    def set_dirichlet_bc(self, gd, uh, threshold=None, q=None, method=None):
        """
        @brief 设置狄利克雷边界条件，使用边界上的 L2 投影
        """
        p = self.p
        mesh = self.mesh
        bspace = self.bspace
        isbdFace = mesh.boundary_face_flag()
        face2dof = self.dof.face_to_dof()[isbdFace]
        fm = mesh.entity_measure('face')[isbdFace]
        gdof = self.number_of_global_dofs()
        n = mesh.edge_unit_normal()[isbdFace]
        # Bernstein 空间的单位质量矩阵
        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = 3*bspace.basis(bcs,p=p)
        M = bm.einsum("cql, cqm, q->lm", bphi, bphi, ws)
        Minv = bm.linalg.inv(M)
        Minv = Minv*fm[:,None,None]

        points = mesh.bc_to_point(bcs)[isbdFace]
        gDval = -gd(points, n) 
        g = -bm.einsum('cql, cq,q->cl', bphi, gDval,ws)
        
        #uh[edge2dof] = bm.einsum('cl, clm->cm', g, Minv) # (NC, ldof)
        uh = bm.set_at(uh,(face2dof),bm.einsum('cl, clm->cm', g, Minv))
        # 边界自由度
        isDDof = bm.zeros(gdof, device=self.device, dtype=bm.bool)
        isDDof[face2dof] = True
        return uh,isDDof

    boundary_interpolate = set_dirichlet_bc

    def show_basis(self, fig, index=0):
        """
        Plot quvier graph for every basis in a fig object
        """

        p = self.p
        mesh = self.mesh
        multiIndex = self.mesh.multi_index_matrix(p, 3)

        ldof = self.number_of_local_dofs()

        bcs = multiIndex(4)/4
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)
        if p == 0:
            m = 2
            n = 2
        elif p == 1:
            m = 5
            n = 3
        elif p == 2:
            m = 6
            n = 6
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i+1, projection='3d')
            mesh.add_plot(axes)
            node = ps[:, index, :]
            v = phi[:, index, i, :]
            l = bm.max(bm.sqrt(bm.sum(v**2, axis=-1)))
            v /=l
            axes.quiver(
                    node[:, 0], node[:, 1], node[:, 2], 
                    v[:, 0], v[:, 1], v[:, 2], length=1)
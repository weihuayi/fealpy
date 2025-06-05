
from typing import Union, TypeVar, Generic, Callable,Optional

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

class RTDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = mesh.multi_index_matrix(p, 2) 
        self.ftype = mesh.ftype 
        self.itype = mesh.itype
        self.device = mesh.device

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+3) 
        elif doftype in {'cell', 2}: # number of dofs inside the cell 
            return (p+1)*p
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
        return bm.arange(NE*edof, device=self.device).reshape(NE, edof)[index]

    def cell_to_dof(self):
        p = self.p
        cldof = self.number_of_local_dofs('cell')
        eldof = self.number_of_local_dofs('edge')
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()

        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        c2e = self.mesh.cell_to_edge()
        edge = self.mesh.entity('edge')
        cell = self.mesh.entity('cell')

        c2d = bm.zeros((NC, ldof),device=self.device, dtype=self.itype)
        #c2d[:, :eldof*3] = e2dof[c2e].reshape(NC, eldof*3)
        c2d = bm.set_at(c2d, (slice(None), slice(None, eldof*3)), e2dof[c2e].reshape(NC, eldof*3))
        s = [1, 0, 0]
        for i in range(3):
            flag = cell[:, s[i]] == edge[c2e[:, i], 1]
            #c2d[flag, eldof*i:eldof*(i+1)] = c2d[flag, eldof*i:eldof*(i+1)][:,::-1]
            c2d = bm.set_at(c2d, (flag, slice(eldof*i, eldof*(i+1))), bm.flip(c2d[flag, eldof*i:eldof*(i+1)], axis=-1))
        #c2d[:, eldof*3:] = bm.arange(NE*eldof, gdof).reshape(NC, -1)
        c2d = bm.set_at(c2d, (slice(None), slice(eldof*3,None)),bm.arange(NE*eldof, gdof).reshape(NC, -1))
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
        flag = bm.zeros(gdof, device=self.device, dtype=bm.bool)

        flag[bddof] = True
        return flag

class RaviartThomasFESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.dof = RTDof2d(mesh, p)

        self.bspace = BernsteinFESpace(mesh, p)
        # self.bspace =LagrangeFESpace(mesh, p)
        # self.bspace1 = LagrangeFESpace(mesh, p-1)
        self.cellmeasure = mesh.entity_measure('cell')
        self.qf = self.mesh.quadrature_formula(p+3)
        self.ftype = mesh.ftype
        self.itype = mesh.itype

        #TODO:JAX
        self.device = mesh.device

    @barycentric
    def basis(self, bcs, index=_S):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        cldof = self.dof.number_of_local_dofs("cell")
        eldof = self.dof.number_of_local_dofs("edge")
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda()

        gp = bm.zeros_like(glambda,device=self.device)
        # gp[..., 0] = -glambda[..., 1]
        # gp[..., 1] =  glambda[..., 0]
        gp = bm.set_at(gp,(...,0), -glambda[..., 1])
        gp = bm.set_at(gp,(...,1), glambda[..., 0])


        ledge = mesh.localEdge

        c2esign = mesh.cell_to_face_sign() #(NC, 3, 2)

        l = bm.zeros((3, )+bcs[None, :, 0, None, None].shape, device=self.device, dtype=self.ftype)
        # l[0] = bc[..., 0, None, None, None]
        # l[1] = bc[..., 1, None, None, None]
        # l[2] = bc[..., 2, None, None, None] #(NQ, NC, ldof, 2)
        l = bm.set_at(l, (0), bcs[None, :, 0,  None, None])
        l = bm.set_at(l, (1), bcs[None, :, 1,  None, None])
        l = bm.set_at(l, (2), bcs[None, :, 2,  None, None])

        # edge basis
        phi = self.bspace.basis(bcs,p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 2)
        val = bm.zeros((NC, )+bcs.shape[:-1]+(ldof, 2), device=self.device, dtype=self.ftype)
        for i in range(3):
            phie = phi[:, :, multiIndex[:, i]==0] #(NQ, NC, eldof)
            c2esi = c2esign[:, i] #(NC, )
            v = l[ledge[i, 0]]*gp[:, None, ledge[i, 1], None] - l[ledge[i, 1]]*gp[:,None ,ledge[i, 0], None]
            #v[..., ~c2esi, :, :] *= -1
            #val[..., eldof*i:eldof*(i+1), :] = phie[..., None]*v
            v = bm.set_at(v, (~c2esi, slice(None), slice(None), slice(None)), -v[~c2esi, :, :, :])
            val = bm.set_at(val, (..., slice(eldof*i, eldof*(i+1)), slice(None)), phie[..., None]*v)
        
        # cell basis
        if(p > 0):
            phi1 = self.bspace.basis(bcs, p=p-1) #(NQ, NC, cldof)
            v0 = l[2]*(l[0]*gp[:, None, 1, None] - l[1]*gp[:, None, 0, None]) #(NQ, NC, ldof, 2)
            v1 = l[0]*(l[1]*gp[:, None, 2, None] - l[2]*gp[:, None, 1, None])

            # val[..., eldof*3:eldof*3+cldof//2, :] = v0*phi[..., None] 
            # val[..., eldof*3+cldof//2:, :] = v1*phi[..., None] 

            val = bm.set_at(val,(..., slice(eldof*3,eldof*3+cldof//2), slice(None)), v0*phi1[..., None])
            val = bm.set_at(val,(..., slice(eldof*3+cldof//2, None), slice(None)), v1*phi1[..., None])
        return val[index]
    
    def cross2d(self, a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    
    @barycentric
    def div_basis(self, bcs, index=_S):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        cldof = self.dof.number_of_local_dofs("cell")
        eldof = self.dof.number_of_local_dofs("edge")
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda()

        gp = bm.zeros_like(glambda)
        # gp[..., 0] = -glambda[..., 1]
        # gp[..., 1] =  glambda[..., 0]
        gp = bm.set_at(gp, (...,0), -glambda[..., 1])
        gp = bm.set_at(gp, (...,1), glambda[..., 0])

        ledge = mesh.localEdge

        c2esign = mesh.cell_to_face_sign() #(NC, 3, 2)

        l = bm.zeros((3, )+bcs[None ,: , 0, None, None].shape, device=self.device, dtype=self.ftype)
        # l[0] = bc[..., 0, None, None, None]
        # l[1] = bc[..., 1, None, None, None]
        # l[2] = bc[..., 2, None, None, None] #(NQ, NC, ldof, 2)
        l = bm.set_at(l, (0), bcs[None, :, 0,  None, None])
        l = bm.set_at(l, (1), bcs[None, :, 1,  None, None])
        l = bm.set_at(l, (2), bcs[None, :, 2,  None, None])
        # edge basis
        phi = self.bspace.basis(bcs)
        gphi = self.bspace.grad_basis(bcs)
        multiIndex = self.mesh.multi_index_matrix(p, 2)
        val = bm.zeros((NC,)+bcs.shape[:-1]+(ldof, ), device=self.device, dtype=self.ftype)
        for i in range(3):
            phie = phi[..., multiIndex[:, i]==0] #(NQ, NC, eldof)
            gphie = gphi[..., multiIndex[:, i]==0, :] #(NQ, NC, eldof, 2)
            c2esi = c2esign[:, i] #(NC, )
            v = l[ledge[i, 0]]*gp[:, None, ledge[i, 1], None] - l[ledge[i, 1]]*gp[:,None, ledge[i, 0], None]
            v = bm.set_at(v,(~c2esi, slice(None),slice(None), slice(None)), -v[~c2esi, :, :, :]) #(NQ, NC, eldof, 2)
            dv = -2*self.cross2d(glambda[:, ledge[i, 0]], glambda[:, ledge[i, 1]]) #(NC, )
            dv[~c2esi] *= -1
            #val[..., eldof*i:eldof*(i+1)] = phie*dv[:, None] + bm.sum(gphie*v, axis=-1)
            val = bm.set_at(val, (..., slice(eldof*i, eldof*(i+1))), phie*dv[:, None, None] + bm.sum(gphie*v, axis=-1)) #(NC, ldof)

        # cell basis
        if(p > 0):
            phi =  self.bspace.basis(bcs, p=p-1) #(NC1, NC, cldof)
            gphi = self.bspace.grad_basis(bcs, p=p-1)# (NC,NQ,cldof,2)


            w0 = l[0]*gp[:, None, 1, None] - l[1]*gp[:, None, 0, None]#(NQ, NC, ldof1, 2)
            w1 = l[1]*gp[:, None, 2, None] - l[2]*gp[:, None, 1, None]

            dw0 = -2*self.cross2d(glambda[:, 0, None], glambda[:, 1, None]) #(NC,1)
            dw1 = -2*self.cross2d(glambda[:, 1, None], glambda[:, 2, None])

            v0 = l[2]*w0 #(NQ, NC, ldof1, 2)
            v1 = l[0]*w1

            dv0 = bm.sum(glambda[:,None, 2, None]*w0, axis=-1) + l[2, ..., 0]*dw0[:, None] #(NQ, NC, ldof)

            dv1 = bm.sum(glambda[:,None, 0, None]*w1, axis=-1) + l[0, ..., 0]*dw1[:, None]

            # val[..., eldof*3:eldof*3+cldof//2] = bm.sum(gphi*v0, axis=-1) + phi*dv0 
            # val[..., eldof*3+cldof//2:] =  bm.sum(gphi*v1, axis=-1) + phi*dv1
            val = bm.set_at(val,(...,slice(eldof*3,eldof*3+cldof//2)),bm.sum(gphi*v0, axis=-1) + phi*dv0)
            val = bm.set_at(val,(...,slice(eldof*3+cldof//2,None)),bm.sum(gphi*v1, axis=-1) + phi*dv1)
        return val[index]

    def edge_basis(self, bcs, index=_S):
        en = self.mesh.edge_normal(index=index)
        en /= bm.sum(en**2, axis=1)[:, None]

        ephi = self.bspace.basis(bcs) #(NE, NQ, eldof)
        val = ephi[..., None] * en[:, None, None,:]
        return -val
    
    def is_boundary_dof(self, threshold=None, method=None):
        return self.dof.is_boundary_dof()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

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
    def grad_value(self, uh, bcs, index=_S):
        pass

    @barycentric
    def edge_value(self, uh, bcs, index=_S):
        pass

    @barycentric
    def face_value(self, uh, bcs, index=_S):
        pass


    def L2_error(self, u, uh):
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
    
    def set_dirichlet_bc(self, gd, uh, threshold=None, q=None, method=None):
        """
        @brief 设置狄利克雷边界条件，使用边界上的 L2 投影
        """
        p = self.p
        mesh = self.mesh
        bspace = self.bspace
        isbdFace = mesh.boundary_face_flag()
        edge2dof = self.dof.edge_to_dof()[isbdFace]
        fm = mesh.entity_measure('face')[isbdFace]
        gdof = self.number_of_global_dofs()
        n = mesh.edge_unit_normal()[isbdFace]
        # Bernstein 空间的单位质量矩阵
        qf = self.mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = bspace.basis(bcs)
        M = bm.einsum("cql, cqm, q->lm", bphi, bphi, ws)
        Minv = bm.linalg.inv(M)
        Minv = Minv*fm[:,None,None]

        points = mesh.bc_to_point(bcs)[isbdFace]
        gDval = -gd(points, n) 
        g = -bm.einsum('cql, cq,q->cl', bphi, gDval,ws)
        
        #uh[edge2dof] = bm.einsum('cl, clm->cm', g, Minv) # (NC, ldof)
        uh = bm.set_at(uh,(edge2dof),bm.einsum('cl, clm->cm', g, Minv))
        # 边界自由度
        isDDof = bm.zeros(gdof, dtype=bm.bool)
        isDDof[edge2dof] = True
        return uh,isDDof

    boundary_interpolate = set_dirichlet_bc
    def project(self, u):
        return self.interpolation(u)

    def interpolation(self, u):
        p = self.p
        mesh = self.mesh

        uh = self.function()
        edge2dof = self.dof.edge_to_dof() 
        en = mesh.edge_unit_normal()
        fm = mesh.entity_measure('face')
        qf = mesh.quadrature_formula(p+3, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        def f0(bc):
            ps = mesh.bc_to_point(bc)
            return -bm.einsum('ijk, ik->ij', u(ps), en)
        uh[edge2dof.reshape(-1)] = bm.einsum('c, i, ci->c', fm, ws, f0(bcs))

        if p >= 1:
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()
            edof = self.number_of_local_dofs('edge')
            idof = self.number_of_local_dofs('cell') # dofs inside the cell 
            cell2dof = NE*edof+ bm.arange(NC*idof).reshape(NC, idof)
            def f1(bc):
                ps = mesh.bc_to_point(bc)
                return u(ps)
            fm = mesh.entity_measure('face')
            qf = mesh.quadrature_formula(p+3, 'face')
            bcs, ws = qf.get_quadrature_points_and_weights()
            val = bm.einsum('c, i, ci->c', fm, ws, f1(bcs))
            uh[cell2dof[:, 0:idof//2]] = val[:, 0, :] 
            uh[cell2dof[:, idof//2:]] = val[:, 1, :]
        return uh
    
    def show_basis(self, fig, index=0, box=None):
        """
        Plot quvier graph for every basis in a fig object
        """
        multiIndex = self.mesh.multi_index_matrix(p, 2)
        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multiIndex(10)/10
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)

        if p == 0:
            m = 1
            n = 3
        elif p == 1:
            m = 4 
            n = 2 
        elif p == 2:
            m = 5 
            n = 3 
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i+1)
            mesh.add_plot(axes, box=box)
            node = ps[:, index, :]
            uv = phi[:, index, i, :]
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], 
                    units='xy')


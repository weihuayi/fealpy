import numpy as np
from fealpy.functionspace.femdof import multi_index_matrix2d, multi_index_matrix3d

from fealpy.decorator import cartesian, barycentric

from fealpy.functionspace.Function import Function
from fealpy.quadrature import FEMeshIntegralAlg

from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import csr_matrix

from fealpy.functionspace import LagrangeFiniteElementSpace


class NedelecDof():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiindex2 = multi_index_matrix2d(p)
        self.multiindex3 = multi_index_matrix3d(p)

    def edge_to_local_face_dof(self):
        multiindex = self.multiindex2
        ldof = self.number_of_local_dofs("faceall")

        eldof = self.number_of_local_dofs('edge')
        e2ld = np.zeros((3, eldof), dtype=np.int_)

        e2ld[0], = np.where(multiindex[:, 0]==0)
        e2ld[0][0] += ldof//2
        e2ld[0][-1] += ldof//2

        e2ld[1] = np.where(multiindex[:, 1]==0)[0][::-1]
        e2ld[1][-1] += ldof//2

        e2ld[2], = np.where(multiindex[:, 2]==0)
        return e2ld

    def face_to_local_dof(self):
        multiindex = self.multiindex3
        ldof = self.number_of_local_dofs()

        fdof = self.number_of_local_dofs('faceall')
        f2ld = np.zeros((4, fdof), dtype=np.int_)
        eldof = self.edge_to_local_face_dof()
        nldof = np.array([[eldof[(i+1)%3, 0], eldof[(i+2)%3, -1]] for i in range(3)])

        localFace = np.array([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)], dtype=np.int_)
        sdof = fdof//2

        for i in range(4):
            flag = localFace[i] > i  
            f2ld[i, :sdof] = np.where(multiindex[:, i]==0)[0]
            f2ld[i, sdof:] = f2ld[i, :sdof] + ldof//3
            f2ld[i, eldof[flag, 1:-1]+sdof] += ldof//3
            f2ld[i, nldof[flag]] += ldof//3
        return f2ld

    def cell_to_dof(self):
        face = self.mesh.entity("face")
        cell = self.mesh.entity("cell")

        multiindex = self.multiindex2.T #(3, ldof)
        cell2face = self.mesh.ds.cell_to_face()
        localFace = np.array([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)],
                dtype=np.int_)

        f2d = self.face_to_dof()
        f2ld = self.face_to_local_dof()
        e2lfd = self.edge_to_local_face_dof()
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        cdof = self.number_of_local_dofs('cell')
        fdof = self.number_of_local_dofs('faceall')

        isndof = np.zeros(ldof, dtype=np.bool_)
        isndof[f2ld] = True

        c2d = np.zeros((NC, ldof), dtype=np.int_)
        idx = np.zeros((NC, 3), dtype=np.int_)
        fe = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int_) #局部边
        for i in range(4):
            fi = face[cell2face[:, i]] #(NC, 3)
            fj = cell[:, localFace[i]]
            _, idx[:, 0] = np.where(fj==fi[:, 0, None])
            _, idx[:, 1] = np.where(fj==fi[:, 1, None])
            _, idx[:, 2] = np.where(fj==fi[:, 2, None])
            k = multiindex[idx[:, 1]] + multiindex[idx[:, 2]] #(NC, fdof)
            didx = k*(k+1)//2+multiindex[idx[:, 2]]
            c2d[:, f2ld[i, :fdof//2]] = f2d[cell2face[:, [i]], didx]
            c2d[:, f2ld[i, fdof//2:]] = f2d[cell2face[:, [i]], didx+fdof//2]

            idx = np.argsort(idx, axis=1)
            # 顶点的自由度可能需要交换
            for j in range(3):
                flag = np.sum(idx[:, fe[fe[j, 0]]]-idx[:, fe[fe[j, 1]]], axis=-1)>0
                tmp = c2d[flag, f2ld[i, e2lfd[(j+1)%3, -1]]].copy()
                c2d[flag, f2ld[i, e2lfd[(j+1)%3, -1]]] = c2d[flag, f2ld[i, e2lfd[(j+2)%3, 0]]]
                c2d[flag, f2ld[i, e2lfd[(j+2)%3, 0]]] = tmp

        c2d[:, ~isndof] = np.arange(gdof-NC*cdof, gdof).reshape(NC, cdof)
        return c2d

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

        istdof = np.zeros(ldof, dtype=np.bool_)
        istdof[e2ldof] = True

        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        f2e = self.mesh.ds.face_to_edge()

        f2esign = np.zeros((NF, 3), dtype=np.bool_)
        f2esign = edge[f2e, 0]==face[:, [1, 2, 0]]

        f2d = np.zeros((NF, ldof), dtype=np.int_)
        # c2d[:, e2ldof] : (NC, 3, p+1), e2dof[c2e] : (NC, 3, p+1)
        tmp = e2dof[f2e]
        tmp[~f2esign] = tmp[~f2esign, ::-1]
        f2d[:, e2ldof] = tmp 
        f2d[:, ~istdof] = np.arange(NE*edof, NE*edof+NF*fdof).reshape(NF, -1)
        return f2d

    def edge_to_dof(self):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge')
        return np.arange(NE*edof).reshape(NE, edof)

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

class SecondNedelecFiniteElementSpace3d():
    def __init__(self, mesh, p, space=None, q = None):
        self.p = p
        self.mesh = mesh
        self.dof = NedelecDof(mesh, p)

        if space is None:
            self.lspace = LagrangeFiniteElementSpace(mesh, p)
        else:
            self.lspace = space

        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(mesh, p+2, cellmeasure=self.cellmeasure)
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

        bval = self.lspace.basis(bc) #(NQ, NC, ldof//3)
        c2v = np.broadcast_to(c2v, val.shape)

        val[..., :ldof//3, :] = bval[..., None]*c2v[..., :ldof//3, :]
        val[..., ldof//3 : 2*(ldof//3):, :] = bval[..., None]*c2v[..., ldof//3:2*(ldof//3), :]
        val[..., 2*(ldof//3):, :] = bval[..., None]*c2v[..., 2*(ldof//3):, :]
        return val

    @barycentric
    def face_basis(self, bc, index=np.s_[:]):
        mesh = self.mesh
        GD = mesh.geo_dimension()
        ldof = (p+1)*(p+2) 

        f2v = self.face_basis_vector(index=index)#(NF, ldof, GD)
        NF = len(f2v)
        
        shape = bc.shape[:-1]
        val = np.zeros(shape+(NF, ldof, GD), dtype=np.float_)

        bval = self.lspace.basis(bc) #(NQ, NC, ldof//3)
        f2v = np.broadcast_to(f2v, val.shape)

        val[..., :ldof//2, :] = bval[..., None]*f2v[..., :ldof//2, :]
        val[..., ldof//2:, :] = bval[..., None]*f2v[..., ldof//2:, :]
        return val

    def face_basis_vector(self, index=np.s_[:]):
        p = self.p
        mesh = self.mesh
        GD = mesh.geo_dimension()
        ldof = (p+1)*(p+2) 
        e2ldof = self.dof.edge_to_local_face_dof()

        f2e = mesh.ds.face_to_edge()[index]
        f2n = mesh.face_unit_normal()[index] #(NF, 3)
        e2t = mesh.edge_unit_tangent()[f2e] #(NF, 3, 3)
        e2n = -np.cross(f2n[:, None], e2t) #(NF, 3, 3)

        NF = len(f2e)

        f2v = np.zeros((NF, ldof, GD), dtype=np.float_)
        f2v[:, :ldof//2] = e2t[:, 0, None] #(NF, ldof//2, 3)
        f2v[:, ldof//2:] = e2n[:, 0, None]

        #f2v[:, e2ldof[:, 1:-1]] : (NF, 3, p-1, GD), e2t: (NF, 3, 3)
        f2v[:, e2ldof[:, 1:-1]] = e2t[:, :, None]
        f2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[:, :, None, :]

        #f2v[:, e2ldof[0, 0]] : (NC, GD), f2e[e2t[0]] : (NC, GD)
        #注意: e2ldof[0, 0] 不是 0
        f2v[:, e2ldof[0, 0]] = e2n[:, 2]/(np.sum(e2n[:, 2]*e2t[:, 0], axis=-1)[:, None]) 
        f2v[:, e2ldof[0, -1]] = e2n[:, 1]/(np.sum(e2n[:, 1]*e2t[:, 0], axis=-1)[:, None]) 
        f2v[:, e2ldof[1, 0]] = e2n[:, 0]/(np.sum(e2n[:, 0]*e2t[:, 1], axis=-1)[:, None]) 
        f2v[:, e2ldof[1, -1]] = e2n[:, 2]/(np.sum(e2n[:, 2]*e2t[:, 1], axis=-1)[:, None]) 
        f2v[:, e2ldof[2, 0]] = e2n[:, 1]/(np.sum(e2n[:, 1]*e2t[:, 2], axis=-1)[:, None]) 
        f2v[:, e2ldof[2, -1]] = e2n[:, 0]/(np.sum(e2n[:, 0]*e2t[:, 2], axis=-1)[:, None]) 
        return f2v

    def face_dof_vector(self, index=np.s_[:]):
        p = self.p
        mesh = self.mesh
        GD = mesh.geo_dimension()
        ldof = (p+1)*(p+2) 
        e2ldof = self.dof.edge_to_local_face_dof()

        f2e = mesh.ds.face_to_edge()[index]
        f2n = mesh.face_unit_normal()[index] #(NF, 3)
        e2t = mesh.edge_unit_tangent()[f2e] #(NF, 3, 3)
        e2n = -np.cross(f2n[:, None], e2t) #(NF, 3, 3)

        NF = len(f2e)

        f2v = np.zeros((NF, ldof, GD), dtype=np.float_)
        f2v[:, :ldof//2] = e2t[:, 0, None] #(NF, ldof//2, 3)
        f2v[:, ldof//2:] = e2n[:, 0, None]

        #f2v[:, e2ldof[:, 1:-1]] : (NF, 3, p-1, GD), e2t: (NF, 3, 3)
        f2v[:, e2ldof] = e2t[:, :, None]
        f2v[:, e2ldof[:, 1:-1]+ldof//2] = e2n[:, :, None, :]
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
        val = np.zeros((bc.shape[0], )+c2v.shape, dtype=np.float_)

        val[..., :ldof//3, :] = np.cross(sgval, c2v[None, ..., :ldof//3, :])
        val[..., ldof//3:2*(ldof//3), :] = np.cross(sgval, c2v[None, ..., ldof//3:2*(ldof//3), :])
        val[..., 2*(ldof//3):, :] = np.cross(sgval, c2v[None, ..., 2*(ldof//3):, :])
        return val

    def basis_vector(self):
        NC = self.mesh.number_of_cells()

        c2e = self.mesh.ds.cell_to_edge()
        c2f = self.mesh.ds.cell_to_face()
        e2t = self.mesh.edge_unit_tangent()
        f2n = self.mesh.face_unit_normal()
        f2e = self.mesh.ds.face_to_edge()

        lf2f = np.array([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)]) 
        lf2e = np.array([[5, 4, 3], [5, 2, 1], [4, 2, 0], [3, 1, 0]]) #(4, 3)

        e2fdof = self.dof.edge_to_local_face_dof() #(3, p+1) 面上, 在边界的自由度 
        f2ldof = self.dof.face_to_local_dof() # 单元上, 在面上的自由度
        ldof = self.dof.number_of_local_dofs() 
        fdof = self.dof.number_of_local_dofs('faceall')

        bv = np.zeros((NC, ldof, 3), dtype=np.float)

        # 内部向量
        bv[:, :ldof//3, 0] = 1
        bv[:, ldof//3:2*(ldof//3), 1] = 1
        bv[:, -ldof//3:, 2] = 1

        # 面内部的标量自由度
        fDofIdx, = np.where(np.all(self.dof.multiindex2!=0, axis=-1)) 
        n2fe = np.array([[(i+1)%3, (i+2)%3] for i in range(3)])
        n2dof = e2fdof[n2fe[:, ::-1], [[0, -1]]] 

        sdof = fdof//2
        for i in range(4):
            # 面内部的自由度
            bv[:, f2ldof[i, fDofIdx]] = e2t[f2e[c2f[:, i], 0], None] 
            bv[:, f2ldof[i, fDofIdx+sdof]] = np.cross(e2t[f2e[c2f[:, i], 0],
                None], f2n[c2f[:, i], None])
            bv[:, f2ldof[i, fDofIdx] + 2*(ldof//3)] = f2n[c2f[:, i], None]

            #边上的切向自由度
            bv[:, f2ldof[i, e2fdof[:, 1:-1]]] = e2t[c2e[:, lf2e[i]], None]
            # 与边垂直的自由度
            # bv[:, f2ldof[i, e2ldof[:, 1:-1]+sdof]] : (NC, 3, p-1, 3)
            f2fn = f2n[c2f[:, lf2f[i]], None]
            f2ft = np.cross(e2t[c2e[:, lf2e[i]], None], f2n[c2f[:, i], None, None])
            bv[:, f2ldof[i, e2fdof[:, 1:-1]+sdof]] = f2fn/(np.sum(f2fn*f2ft,
                axis=-1))[..., None]
            # 顶点上的自由度
            # bv[:, f2ldof[i, n2dof]] (NC, 3, 2, 3)
            # f2n[c2f[:, n2fe]] (NC, 3, 2, 3)
            tmp0 = f2n[c2f[:, lf2f[i, n2fe]]]
            tmp1 = e2t[c2e[:, lf2e[i, n2fe[:, ::-1]]]]
            bv[:, f2ldof[i, n2dof]] = tmp0/(np.sum(tmp0*tmp1, axis=-1))[..., None]
        return bv

    def dof_vector(self):
        bv = self.basis_vector()
        NC = bv.shape[0]
        dv = np.linalg.inv(bv.reshape(NC, 3, -1, 3).swapaxes(1, 2)).swapaxes(-1,
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

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        mass = np.einsum("qclg, qcdg, c, q->cld", phi, phi, cm, ws)

        I = np.broadcast_to(c2d[:, :, None], shape=mass.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=mass.shape)
        M = csr_matrix((mass.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def curl_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        cphi = self.curl_basis(bcs) #(NQ, NC, ldof, GD)
        A = np.einsum("qclg, qcdg, c, q->cld", cphi, cphi, cm, ws) #(NC, ldof, ldof)

        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return B

    def projection(self, f, method="L2"):
        M = self.mass_matrix()
        b = self.source_vector(f)
        x, _ = cg(M, b, tol="1e-12")
        return self.function(array=x)

    def function(self, dim=None, array=None, dtype=np.float_):
        if array is None:
            gdof = self.dof.number_of_global_dofs()
            array = np.zeros(gdof, dtype=np.float_)
        return Function(self, dim=dim, array=array, coordtype='barycentric', dtype=dtype)

    def interplation(self, f):
        dv = self.dof_vector() #(NC, ldof, 3)
        ldof = self.dof.number_of_local_dofs()
        point = self.lspace.interpolation_points() #(NC, ldof//3, 3)
        lcell2dof = self.lspace.dof.cell_to_dof()
        cell2dof = self.dof.cell_to_dof()
        fval = f(point)[lcell2dof]

        fh = self.function()
        fh[cell2dof[:, :ldof//3]] = np.sum(fval*dv[:, :ldof//3], axis=-1)
        fh[cell2dof[:, ldof//3:2*ldof//3]] = np.sum(fval*dv[:, ldof//3:2*ldof//3], axis=-1)
        fh[cell2dof[:, 2*ldof//3:]] = np.sum(fval*dv[:, 2*ldof//3:], axis=-1)
        return fh

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        p = self.p
        mesh = self.mesh
        ldof = (p+1)*(p+2) 
        gdof = self.number_of_global_dofs()
       
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()

        face2dof = self.dof.face_to_dof()[index]
        f2v = self.face_dof_vector(index=index) #(NF, ldof, 3)

        bcs = self.lspace.multi_index_matrix[2](p)/p
        point = mesh.bc_to_point(bcs, index=index).swapaxes(0, 1)

        gval = gD(point) #(NF, ldof//2, 3)

        uh[face2dof[:, :ldof//2]] = np.sum(gval*f2v[:, :ldof//2], axis=-1)
        uh[face2dof[:, ldof//2:]] = np.sum(gval*f2v[:, ldof//2:], axis=-1)

        isDDof = np.zeros(gdof, dtype=np.bool_)
        isDDof[face2dof] = True
        return isDDof

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
        val = np.einsum("cl, ...clk->...ck", uh[c2d], phi)
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
        errval = np.sum((uval-uhval)*(uval-uhval), axis=-1)#(NQ, NC)
        val = np.einsum("qc, q, c->", errval, ws, cm)
        return np.sqrt(val)

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
        errcval = np.sum((cuval-cuhval)**2, axis=-1)#(NQ, NC)
        val = np.einsum("qc, q, c->", errcval, ws, cm)
        return np.sqrt(val)

    def source_vector(self, f):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
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
        val = np.einsum("qcg, qclg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = np.zeros(gdof, dtype=np.float_)
        np.add.at(vec, c2d, val)
        return vec




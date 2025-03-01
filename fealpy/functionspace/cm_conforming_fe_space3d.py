from typing import Union, TypeVar, Generic, Callable, Optional
import itertools
from urllib.request import noheaders

#from networkx.classes import number_of_nodes

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from fealpy.functionspace.bernstein_fe_space import BernsteinFESpace
from .functional import symmetry_span_array, symmetry_index, span_array
from scipy.special import factorial, comb
from fealpy.decorator import barycentric


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class CmConformingFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh:_MT, p: int, m: int): 
        assert(p>8*m)
        self.mesh = mesh
        self.p = p
        self.m = m
        self.bspace = BernsteinFESpace(mesh, p)
        self.device = mesh.device

        #self.device = mesh.device
        self.ikwargs = bm.context(mesh.cell)
        self.fkwargs = bm.context(mesh.node)
        self.ftype = mesh.ftype
        self.itype = mesh.itype


        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.multiIndex = mesh.multi_index_matrix(p, 3)
        self.dof_index = {}
        self.get_dof_index()
 
        #self.isCornerNode = isCornerNode
        #self.isCornerNode, self.isCornerEdge,_,_,_ = self.get_corner()
        self.coeff = self.coefficient_matrix()
   
    def get_dof_index(self):
        p = self.p
        m = self.m
        midx = self.multiIndex
        ldof = midx.shape[0]

        isn_cell_dof = bm.zeros(ldof, dtype=bm.bool, device=self.device)

        node_dof_index = []
        for i in range(4):
            a = []
            for r in range(4*m+1):
                Dvr = midx[:, i]==p-r
                isn_cell_dof = isn_cell_dof | Dvr
                a.append(bm.where(bm.array(Dvr))[0])
            node_dof_index.append(a)

        locEdge = bm.array([[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]],
                           **self.ikwargs)
        dualEdge = bm.array([[2, 3],[1, 3],[1, 2],[0, 3],[0, 2],[0, 1]],
                            **self.ikwargs)
        edge_dof_index = []  
        for i in range(6):
            a = []
            for r in range(2*m+1):
                aa = []
                for j in range(r+1):
                    Derj = (bm.sum(bm.array(midx[:, locEdge[i]]), axis=-1)==p-r) & (midx[:, dualEdge[i, 1]]==j) & (~isn_cell_dof)
                    isn_cell_dof = isn_cell_dof | Derj
                    aa.append(bm.where(Derj)[0])
                a.append(aa)
            edge_dof_index.append(a)

        face_dof_index = []
        for i in range(4):
            a = []
            for r in range(m+1):
                Dfr = (bm.array(midx[:, i]==r)) & (~isn_cell_dof)
                isn_cell_dof = isn_cell_dof | Dfr
                a.append(bm.where(Dfr)[0])
            face_dof_index.append(a)

        
        all_node_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in node_dof_index for item in sublist])
        all_edge_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in edge_dof_index for item in sublist])
        all_face_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in face_dof_index for item in sublist])

        self.dof_index["node"] = node_dof_index
        self.dof_index["edge"] = edge_dof_index
        self.dof_index["face"] = face_dof_index
        self.dof_index["cell"] = bm.where(isn_cell_dof==0)[0]
        self.dof_index["all"] = bm.concatenate((all_node_dof_index,
                                               all_edge_dof_index,
                                               all_face_dof_index,
                                               self.dof_index["cell"]),**self.ikwargs)

    def number_of_local_dofs(self, etype, p=None) -> int: #TODO:去掉etype 2d同样
        p = self.p if p is None else p
        if etype=="cell":
            return (p+1)*(p+2)*(p+3)//6
    def number_of_internal_dofs(self, etype, p=None, m=None) -> int:
        p = self.p if p is None else p
        m = self.m if m is None else m
        if etype=='cell':
            return len(self.dof_index["cell"])
        if etype=="edge":
            N = 0
            for edof_r in self.dof_index["edge"][0]:
                for edof_r_j in edof_r:
                    N += len(edof_r_j)
            return N 
        if etype=="face":
            N = 0
            for fdof_r in self.dof_index["face"][0]:
                N += len(fdof_r)
            return N 
        if etype=='node':
            return (4*m+1)*(4*m+2)*(4*m+3)//6
    def number_of_global_dofs(self) -> int:
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        ndof = self.number_of_internal_dofs('node')
        eidof = self.number_of_internal_dofs('edge')
        fidof = self.number_of_internal_dofs('face')
        cidof = self.number_of_internal_dofs('cell')
        return NN*ndof +eidof*NE + fidof*NF + cidof*NC

    def node_to_dof(self):
        m = self.m
        mesh = self.mesh
        ndof = self.number_of_internal_dofs('node')
        NN = mesh.number_of_nodes()
        n2d = bm.arange(NN*ndof, **self.ikwargs).reshape(NN, ndof)
        return n2d

    def edge_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge') 

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        e2id = bm.arange(NN*ndof, NN*ndof + NE*eidof, **self.ikwargs).reshape(NE, eidof)
        return e2id

    def face_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge') 
        fidof = self.number_of_internal_dofs('face') 

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        Ndof = NN*ndof + NE*eidof
        f2id = bm.arange(Ndof, Ndof + NF*fidof, **self.ikwargs).reshape(NF, fidof)
        return f2id

    def cell_to_internal_dof(self):
        p = self.p
        m = self.m
        ldof = self.number_of_local_dofs('cell')
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge') 
        fidof = self.number_of_internal_dofs('face') 
        cidof = self.number_of_internal_dofs('cell') 

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        Ndof = NN*ndof + NE*eidof + NF*fidof
        c2id = bm.arange(Ndof, Ndof + NC*cidof, **self.ikwargs).reshape(NC, cidof)
        return c2id

    def cell_to_dof(self):
        p, m = self.p, self.m
        mesh = self.mesh

        ldof = self.number_of_local_dofs('cell')
        ndof = self.number_of_internal_dofs('node')
        eidof = self.number_of_internal_dofs('edge')
        fidof = self.number_of_internal_dofs('face')
        cidof = self.number_of_internal_dofs('cell')

        n2d = self.node_to_dof()
        e2id = self.edge_to_internal_dof()
        f2id = self.face_to_internal_dof()
        c2id = self.cell_to_internal_dof()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()

        c2dof = bm.zeros((NC, ldof), **self.ikwargs)
        ## node
        for v in range(4):
            c2dof[:, v*ndof:(v+1)*ndof] = n2d[cell[:, v]]

        ## edge
        localEdge = bm.array([[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]],
                             **self.ikwargs)
        for e in range(6):
            N = ndof*4 + eidof*e
            c2dof[:, N:N+eidof] = e2id[c2e[:, e]]
            flag = edge[c2e[:, e], 0] != cell[:, localEdge[e, 0]]
            n0, n1 = 0, p-8*m-1
            for r in range(2*m+1):
                for j in range(r+1):
                    #c2dof[flag, N+n0:N+n0+n1] = bm.flip(c2dof[flag, N+n0:N+n0+n1], axis=-1)
                    c2dof = bm.set_at(c2dof, (flag, slice(N+n0,N+n0+n1)), bm.flip(c2dof[flag, N+n0:N+n0+n1], axis=-1))
                    n0 +=n1
                n1 += 1
        ## face
        fdof_index = self.dof_index["face"]
        perms = bm.array(list(itertools.permutations([0,1,2])))
        locFace = bm.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], **self.ikwargs)
        midx2num = lambda a: (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2+a[:, 2]

        indices = []
        for r in range(m+1):
            dof_fc2f = midx2num(self.multiIndex[fdof_index[0][r]][:, locFace[0]])
            midx = self.mesh.multi_index_matrix(p-r, 2)
            indices_r = bm.zeros((6, len(dof_fc2f)), **self.ikwargs)
            for i in range(6):
                indices_r[i] = bm.argsort(dof_fc2f)[bm.argsort(bm.argsort( midx2num( self.multiIndex[ fdof_index[0][r]][:, locFace[0]][:, perms[i]])))]
                indices_r[i] = bm.argsort(bm.argsort(midx2num(midx[dof_fc2f][ :, perms[i]])))
            indices.append(indices_r)
        c2fp = self.mesh.cell_to_face_permutation(locFace=locFace)
        perm2num = lambda a: a[:, 0]*2 +(a[:, 1]>a[:, 2])
        for f in range(4):
            N = ndof*4 +eidof*6 +fidof*f
            #c2dof[:, N:N+fidof]  = f2id[c2f[:, f]]
            c2dof = bm.set_at(c2dof, (slice(None), slice(N, N+fidof)),
                              f2id[c2f[:, f]])
            pnum = perm2num(c2fp[:, f])
            for i in range(6):
                n0 = 0
                flag = pnum == i
                for r in range(m+1):
                    idx = indices[r][i]
                    n1 = idx.shape[0]
                    c2dof[flag, N+n0:N+n0+n1] = c2dof[flag, N+n0:N+n0+n1][:, idx]
                    n0 += n1
        ## cell
        c2dof[:, ldof-cidof:] = c2id
        return c2dof

    def boundary_interpolate(self, gd, uh, threshold=None, method="interp"):
        isDDof = self.is_boundary_dof(threshold=threshold)
        uI = self.interpolation(gd)
        uh[isDDof] = uI[isDDof]
        return uh, isDDof


    def is_boundary_dof(self, threshold=None, method='interp'): #TODO:threshold 未实现
        mesh = self.mesh
        m = self.m
        p = self.p
        multiIndex = self.multiIndex
        iscornernode, isbdedgenode, isbdfacenode, iscorneredge, isbdfaceedge = self.get_corner()
        gdof = self.number_of_global_dofs()

        # 点
        n2d = self.node_to_dof()
        cnidx = n2d[iscornernode]
        enidx = bm.arange(len(multiIndex))[((multiIndex[:, 3]<=m) | (multiIndex[:, 2]<=m)) & (multiIndex[:, 0]>= p-4*m)]
        enidx = n2d[isbdedgenode][:, enidx]
        fnidx = bm.arange(len(multiIndex))[(multiIndex[:, 3] <=m) & (multiIndex[:, 0]>= p-4*m)]
        fnidx = n2d[isbdfacenode][:, fnidx]
        nidx = bm.concatenate([cnidx.flatten(), enidx.flatten(), fnidx.flatten()])

        # 边
        e2id = self.edge_to_internal_dof()
        ceidx = e2id[iscorneredge]
        flag = (multiIndex[:, 0]<p-4*m) & (multiIndex[:, 1]<p-4*m) & (bm.sum(multiIndex[:, 2:], axis=1)<=2*m) & (multiIndex[:, 3]<=m) 
        a = self.dof_index['edge'][0]
        a = bm.concatenate([arr for sublist in a for arr in sublist])
        b = bm.arange(len(multiIndex))[flag]
        indices = [bm.where(a == num)[0][0] for num in b]
        feidx = e2id[isbdfaceedge][:, indices]
        eidx = bm.concatenate([ceidx.flatten(), feidx.flatten()])

        # 面
        isbdface = mesh.boundary_face_flag()
        f2id = self.face_to_internal_dof()
        fidx = f2id[isbdface].flatten()
        idx = bm.concatenate([nidx, eidx, fidx])

        isBdDof = bm.zeros(gdof, dtype=bm.bool, device=self.device)
        isBdDof[idx] = True
        return isBdDof


#        isbdnode = mesh.boundary_node_flag()
#        bdnodedof = self.node_to_dof()[isbdnode]
#        l = (m+1)*(m+2)//2
#        a = self.dof_index["node"]#[0]
#        print('aa',a)
#        idxnode = []
#        d = 0
#        for r in a:
#            l = d+ d+1
#            idx1 = r[:l]
#            d = d+1
#            #print(idx1)
#            idxnode.append(idx1)
#        idxnode = bm.concatenate(idxnode)
#        bdnodedof = bdnodedof[:, idxnode].reshape(-1)
#
#        isbdedge = mesh.boundary_edge_flag()
#        bdedgedof = self.edge_to_internal_dof()[isbdedge]
#        a = self.dof_index["edge"][0]
#        a = bm.concatenate([arr for sublist in a for arr in sublist][:5]) 
#        a = bm.arange(a.shape[0])
#        bdedgedof = bdedgedof[:, a].reshape(-1) 
#
#        # 面
#        isbdface = mesh.boundary_face_flag()
#        bdfacedof = self.face_to_internal_dof()[isbdface].reshape(-1)
#        #print(bdfacedof)
#        isBdDof = bm.zeros(self.number_of_global_dofs(), dtype=bm.bool) 
#        bdidx = bm.concatenate([bdnodedof, bdedgedof, bdfacedof])
#        fidof = self.number_of_internal_dofs('face')
#        #print('2222222',bdidx.shape)
#        isBdDof[bdidx] = True
#        return isBdDof 

#    def is_boundary_dof(self, threshold=None): #TODO:threshold 未实现
#        mesh = self.mesh
#        m = self.m
#        isbdface = mesh.boundary_face_flag()
#        bdfaceidx = bm.arange(isbdface.shape[0])[isbdface]
#        #print(bdfaceidx)
#        c2f = mesh.cell_to_face()
#        multiIndex = self.multiIndex
#        print(multiIndex)
#        nodedof = self.dof_index['node']
#        ndi = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in nodedof for item in sublist])
#
#        print(ndi)
#        print(multiIndex[ndi])
#        #mask = bm.isin(c2f, bdfaceidx)
#        l = lambda x: (x[:,1]+x[:,2]+x[:,3]+2)*(x[:,1]+x[:,2]+x[:,3]+1)*(x[:,1]+x[:,2]+x[:,3])//6 + (x[:,2]+x[:,3]+1)*(x[:,2]+x[:,3])//2 + x[:,3]
#        cell2dof = self.cell_to_dof() 
#        #print(cell2dof)
#        #print(self.dof_index['all'])
#        bdidx = []
#        for f in bdfaceidx:
#            ncell, nu = bm.where(c2f==f)
#            #print(ncell,nu)
#            isbdmul = multiIndex[:, int(nu)]<=m 
#
#            bdmul = multiIndex[isbdmul]
#            #print(bdmul.shape)
#            idx = l(bdmul)
#            #print('idx',idx)
#            idx = bm.argsort(self.dof_index['all'])[idx]
#        
#
#            bdidx.append(cell2dof[ncell, idx])
#            #print(bdidx)
#        bdidx = bm.unique(bm.concatenate(bdidx))
#        print('111111', bdidx.shape)
#        return

    def get_corner(self):
        """
        @brief 获取角点, 角边, 不太对啊
        """
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')

        f2e = mesh.face_to_edge()
        fn  = mesh.face_unit_normal()
        NE  = mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        # 角点
        #isCornerNode = ((bm.abs(node[:, 0]-box[0])<1e-14) | (bm.abs(node[:, 0]-box[1])<1e-14)) & ((bm.abs(node[:, 1]-box[2])<1e-14) | (bm.abs(node[:, 1]-box[3])<1e-14)) & ((bm.abs(node[:, 2]-box[4])<1e-14)| (bm.abs(node[:, 2]-box[5])<1e-14)) 

        # 棱边
        isCornerEdge = bm.zeros(NE, dtype=bm.bool, device=self.device)
        isBdFace = mesh.boundary_face_flag()
        bb = bm.zeros((NE, 3), **self.fkwargs)
        bb[f2e[isBdFace]] = fn[isBdFace, None]
        fn = bm.tile(fn[:, None], (1, 3, 1))
        flag = bm.linalg.norm(bm.cross(bb[f2e[isBdFace]], fn[isBdFace]), axis=-1)>1e-10
        isCornerEdge[f2e[isBdFace][flag]] = True

        # 棱点
        cornernode, num = bm.unique(edge[isCornerEdge].flatten(), return_counts=True)
        isBdEdgeNode = bm.zeros(NN, dtype=bm.bool, device=self.device)
        isBdEdgeNode[cornernode[num==2]] = True

        isCornerNode = bm.zeros(NN, dtype=bm.bool, device=self.device)
        isCornerNode[cornernode[num>2]] = True
        #isBdEdgeNode[edge[isCornerEdge]] = True
        #isBdEdgeNode[isCornerNode] = False

        # 边界面上点
        isBdNode = mesh.boundary_node_flag()
        isBdFaceNode = isBdNode
        isBdFaceNode[isBdEdgeNode | isCornerNode] = False

        # 面上边界边
        isBdFaceEdge = mesh.boundary_edge_flag()
        isBdFaceEdge[isCornerEdge] = False

        return isCornerNode, isBdEdgeNode, isBdFaceNode,  isCornerEdge,isBdFaceEdge 



    def get_frame(self):
        """
        @brief 获取每个点，每条边，每个面的全局坐标系
        """
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()
        f2e = mesh.face_to_edge()

        fn = mesh.face_unit_normal()
        et = mesh.edge_tangent(unit=True)
        isBdNode = mesh.boundary_node_flag()
        isBdEdge = mesh.boundary_edge_flag()
        isBdFace = mesh.boundary_face_flag()


        # face frame
        NF = mesh.number_of_faces()
        face_frame = bm.zeros((NF, 3, 3), **self.fkwargs)
        face_frame[:, 2] = fn
        face_frame[:, 0] = et[f2e[:, 0]] 
        face_frame[:, 1] = bm.cross(face_frame[:, 2], face_frame[:, 0])


        # edge frame
        NE = mesh.number_of_edges()
        edge_frame = bm.zeros((NE, 3, 3), **self.fkwargs)
        edge_frame[:, 0] = et
        edge_frame[f2e, 2] = fn[:, None] 

        # edge frame at boundary
        edge_frame[f2e[isBdFace], 2] = fn[isBdFace, None]
        edge_frame[:, 1] = bm.cross(edge_frame[:, 2], edge_frame[:, 0])

        # node frame 
        NN = mesh.number_of_nodes()
        node_frame = bm.zeros((NN, 3, 3), **self.fkwargs)
        node_frame[:] = bm.eye(3, **self.fkwargs)

        # 边界表面点
        node_frame[face[isBdFace]] = face_frame[isBdFace, None]

        # 边界棱点
        isCornerNode, isbdedgenode, isbdfacenode, iscorneredge, isbdfaceedge = self.get_corner()
        node_frame[edge[iscorneredge]] = edge_frame[iscorneredge, None]

        # 角点
        node_frame[isCornerNode] = bm.eye(3, **self.fkwargs)
#        for i in range(len(f2e)):
#            edge_frame[f2e[i], 2] = fn[i][None,:]
#        for i in range(len(edge[iscorneredge])):
#            node_frame[edge[iscorneredge][i]] = edge_frame[iscorneredge][i][None,:]
#        for i in range(len(face[isBdFace])):
#            node_frame[face[isBdFace][i]] = face_frame[isBdFace][i][None,:]
#        for i in range(len(f2e[isBdFace])):
#            edge_frame[f2e[isBdFace][i], 2] = fn[isBdFace][i][None,:]
#
      
        return node_frame, edge_frame, face_frame



        
    def coefficient_matrix(self):
        p = self.p
        m = self.m
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()
        f2e = mesh.face_to_edge() 

        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        NE = mesh.number_of_edges()
        ldof = self.number_of_local_dofs('cell')

        tem = bm.eye(ldof, **self.fkwargs)
        coeff = bm.tile(tem, (NC, 1, 1)) 

        all_dof_idx = self.dof_index["all"]
        dof2num = bm.zeros(ldof, **self.ikwargs) 
        dof2num[all_dof_idx] = bm.arange(ldof, **self.ikwargs)

        multiIndex = self.multiIndex
        S04m = self.dof_index["node"]
        # node
        for v in range(4):
            flag = bm.ones(4, dtype=bm.bool, device=self.device)
            flag[v] = False # v^*
            S04mv = bm.concatenate(S04m[v])
            for i in S04mv:
                alpha = multiIndex[i][flag]
                r = int(bm.sum(alpha))
                for j in all_dof_idx:
                    gamma = multiIndex[j]
                    if bm.any(alpha-gamma[flag]<0):
                        continue
                    sign = (-1)**(gamma[v]-(p-r)) #\alpha_v = p-r
                    I = dof2num[i]
                    J = dof2num[j]
                    alpha = bm.to_numpy(alpha)
                    flag = bm.to_numpy(flag)
                    gamma = bm.to_numpy(gamma)

                    coeff[:, I, J] = sign * factorial(r)*comb(p,r) * bm.prod(bm.array(comb(alpha, gamma[flag])))
                    alpha = bm.array(alpha)
                    flag = bm.array(flag)
                    gamma = bm.array(gamma)
        ## 测试每个单元顶点的系数矩阵是一样的,与单元无关
        #import numpy as np
        #for i  in range(NC-1):
        #    np.testing.assert_allclose(coeff[i,:105,:105],coeff[i+1,:105,:105], atol=1e-15)
        #    print(i)
        #print(coeff[:, :105, :105])
        
           
        # edge
        glambda = mesh.grad_lambda() # (NC, 4, 3)
        S12m = self.dof_index["edge"]
        locEdge = bm.array([[0 ,1], [0, 2], [0, 3], 
                            [1, 2], [1, 3], [2, 3]], **self.ikwargs)
        dualEdge = bm.array([[2, 3], [1, 3], [1, 2], 
                             [0, 3], [0, 2], [0, 1]], **self.ikwargs)
        for e in range(6):
            ii, jj = locEdge[e]
            kk, ll = dualEdge[e]
            # local normal frame
            nlambdaktl = bm.cross(glambda[:, kk], glambda[:, ll]) # (NC, 3)
            nlambdaktl_l = bm.linalg.norm(nlambdaktl, axis=1)[:, None]**2
            Ni = bm.cross(glambda[:, ll], nlambdaktl)/nlambdaktl_l # (NC, 3)
            Nj = bm.cross(nlambdaktl, glambda[:, kk])/nlambdaktl_l
            S12me = bm.concatenate([bm.concatenate(item) for item in S12m[e]])
            Eij = bm.zeros((NC, 4), **self.fkwargs)
            Eij[:, 0] = bm.sum(glambda[:, ii]*Ni, axis=1)
            Eij[:, 1] = bm.sum(glambda[:, ii]*Nj, axis=1)
            Eij[:, 2] = bm.sum(glambda[:, jj]*Ni, axis=1)
            Eij[:, 3] = bm.sum(glambda[:, jj]*Nj, axis=1)
            for i in S12me:
                alpha = multiIndex[i]
                alpha_ij = alpha[locEdge[e]]
                alpha_kl = alpha[dualEdge[e]] # e^*
                r = int(bm.sum(alpha_kl))
                for j in all_dof_idx:
                    gamma = multiIndex[j] 
                    gamma_ij = gamma[locEdge[e]]
                    gamma_kl = gamma[dualEdge[e]]
                    if bm.any(alpha_kl-gamma_kl<0)|bm.any(gamma_ij-alpha_ij<0):
                        continue
                    c = factorial(r)*comb(p, r)
                    alpha_kl = bm.to_numpy(alpha_kl)
                    gamma_kl = bm.to_numpy(gamma_kl)
                    c *= bm.prod(bm.array(comb(alpha_kl, gamma_kl)))
                    alpha_kl = bm.array(alpha_kl)
                    gamma_kl = bm.array(gamma_kl)

                    I = dof2num[i]
                    J = dof2num[j]
                    coeff[:, I, J] = 0 #这个为什么
                    sigmas = mesh.multi_index_matrix(gamma[ii]-alpha[ii], 1)
                    for sig in sigmas:
                        if bm.any(alpha_kl-sig-gamma_kl<0):
                            continue
                        alpha_kl = bm.to_numpy(alpha_kl)
                        gamma_kl = bm.to_numpy(gamma_kl)
                        sig = bm.to_numpy(sig)

                        cc = c*bm.prod(bm.array(comb(alpha_kl-gamma_kl, sig)))
                        alpha_kl = bm.array(alpha_kl)
                        gamma_kl = bm.array(gamma_kl)
                        sig = bm.array(sig)
                        coeff[:, I, J] += cc*bm.prod(Eij**bm.concatenate([sig, alpha_kl-gamma_kl-sig]), axis=1)
        # face
        fn = mesh.face_unit_normal()
        S2m = self.dof_index["face"]
        locFace = bm.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                           **self.ikwargs)
        for f in range(4):
            ii, jj, kk = locFace[f] 
            ni = fn[c2f[:, f]] # (NC, 3) 
            Fki = bm.zeros((NC, 4), dtype=bm.float64)
            Fki[:, 0] = bm.sum(glambda[:, 0]*ni, axis=-1)
            Fki[:, 1] = bm.sum(glambda[:, 1]*ni, axis=-1)
            Fki[:, 2] = bm.sum(glambda[:, 2]*ni, axis=-1)
            Fki[:, 3] = bm.sum(glambda[:, 3]*ni, axis=-1)
            S2mf = bm.concatenate(S2m[f])

            for i in S2mf:
                alpha = multiIndex[i]
                g_jkl = alpha[locFace[f]]
                r = int(alpha[f]) # |alpha_{f^*}|
                for j in all_dof_idx:
                    gamma = multiIndex[j]
                    beta = gamma-alpha
                    beta[f] += r
                    if bm.any(beta<0):
                        continue
                    beta = bm.to_numpy(beta)
                    c = factorial(r)**2*comb(p, r)/bm.prod(bm.array(factorial(beta)))
                    beta = bm.array(beta)
                    #flag = beta>0
                    val = c*bm.prod(Fki**beta, axis=1)
                    I = dof2num[i]
                    J = dof2num[j]
                    coeff[:, I, J] = val 

        ## 测试每个单元顶点的系数矩阵是一样的,与单元无关
        #import numpy as np
        #for i  in range(NC-1):
        #    np.testing.assert_allclose(coeff[i,:105,:105],coeff[i+1,:105,:105], atol=1e-15)
        #    print(i)
        #print(coeff[:, :105, :105])
        

        coeff = bm.linalg.inv(coeff)
        coeff = bm.transpose(coeff, (0, 2, 1))  

        nframe, eframe, _ = self.get_frame()
        # 全局自由度矩阵
        # node 
        ndof = self.number_of_internal_dofs('node')
        coeff1 = bm.zeros((NC, 4*ndof, 4*ndof), **self.fkwargs)
        for v in range(4):
            j, k, l = locFace[v]
            Nv = bm.zeros((NC, 3, 3), **self.fkwargs) # N_v
            Nv[:, 0] = node[cell[:, j]] - node[cell[:, v]]
            Nv[:, 1] = node[cell[:, k]] - node[cell[:, v]]
            Nv[:, 2] = node[cell[:, l]] - node[cell[:, v]]
            nv = nframe[cell[:, v]] # (NC, 3, 3)
            coeff1[:, ndof*v, ndof*v] = 1
            kk = 1
            for r in range(4*m+1)[1:]:
                symidx, num = symmetry_index(3, r)
                multiidx = mesh.multi_index_matrix(r, 2)

                NSr2 = len(multiidx)
                T = bm.zeros((NC, NSr2, NSr2), **self.fkwargs)
                for iii, alphaa in enumerate(multiidx): 
                    Nv_sym = symmetry_span_array(Nv, alphaa).reshape(-1, 3**r)[:, symidx]
                    for jjj, betaa in enumerate(multiidx):
                        nv_sym = symmetry_span_array(nv, betaa).reshape(-1, 3**r)[:, symidx]
                        T[:, iii, jjj] = bm.sum(Nv_sym*nv_sym*num[None, :], axis=1)
                coeff1[:, ndof*v+kk:ndof*v+kk+NSr2, ndof*v+kk:ndof*v+kk+NSr2] = T
                kk += NSr2
        #coeff[:, :4*ndof] = bm.einsum('cji, cjk->cik',coeff1, coeff[:, :4*ndof])
        coeff[:, :4*ndof] = coeff1.transpose(0,2,1)@coeff[:, :4*ndof]
        del coeff1
        import gc
        gc.collect()

        # edge
        edof = self.number_of_internal_dofs('edge')
        coeff2 = bm.zeros((NC, 6*edof, 6*edof), **self.fkwargs)

        fn = mesh.face_unit_normal()
        et = mesh.edge_tangent(unit=True)
        en = eframe[:, 1:]
        for e in range(6):
            ii, jj = locEdge[e]
            kk, ll = dualEdge[e]
            nlambdaktl = bm.cross(glambda[:, kk], glambda[:, ll]) # (NC, 3)
            nlambdaktl_l = bm.linalg.norm(nlambdaktl, axis=1)[:, None]**2
            Ni = bm.cross(glambda[:, ll], nlambdaktl)/nlambdaktl_l # (NC, 3)
            Nj = bm.cross(nlambdaktl, glambda[:, kk])/nlambdaktl_l

            Ncoef = bm.zeros((NC, 2, 2), **self.fkwargs)
            Ncoef[:, 0, 0] = bm.sum(Ni*en[c2e[:, e], 0], axis=1) 
            Ncoef[:, 0, 1] = bm.sum(Ni*en[c2e[:, e], 1], axis=1)
            Ncoef[:, 1, 0] = bm.sum(Nj*en[c2e[:, e], 0], axis=1)
            Ncoef[:, 1, 1] = bm.sum(Nj*en[c2e[:, e], 1], axis=1)
            edof_idx = self.dof_index["edge"][e]
            if(len(edof_idx[0][0])>0):
                idxidx = dof2num[edof_idx[0][0]]-ndof*4
                coeff2[:, idxidx, idxidx] = 1.0
            for r in range(2*m+1)[1:]:
                symidx, num = symmetry_index(2, r)
                multiidx = mesh.multi_index_matrix(r, 1)
                edof_idxr = edof_idx[r]
                for i in range(r+1):
                    midx = multiidx[i]
                    Ncoef_sym = symmetry_span_array(Ncoef, midx).reshape(-1,
                                                                         2**r)
                    for j in range(r+1):
                        coeff2[:, dof2num[edof_idxr[i]]-ndof*4, dof2num[edof_idxr[j]]-ndof*4] = num[j] * Ncoef_sym[:, symidx[j], None] 
        #coeff[:, 4*ndof:4*ndof+6*edof] = bm.einsum('cji, cjk->cik',coeff2, coeff[:, 4*ndof:4*ndof+6*edof])
        coeff[:, 4*ndof:4*ndof+6*edof] = coeff2.transpose(0,2,1)@coeff[:, 4*ndof:4*ndof+6*edof]
        del coeff2
        gc.collect()
        coeff = coeff[:,:,dof2num]
        return coeff

    def basis(self, bcs, index=_S):
        coeff = self.coeff
        bphi = self.bspace.basis(bcs)
        #return bm.einsum('cil, cql -> cqi', coeff, bphi)[:,:,index]
        return bphi @ (coeff.transpose(0,2,1)[index])

    def grad_m_basis(self, bcs, m):
        coeff = self.coeff
        bgmphi = self.bspace.grad_m_basis(bcs, m)
        return bm.einsum('cil, cqlg -> cqig', coeff, bgmphi)

    @barycentric
    def value(self, uh, bc, index=_S):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        val = bm.einsum('cql, cl->cq', phi, uh[cell2dof])
        return val

    @barycentric
    def grad_m_value(self, uh, bc, m):
        gmphi = self.grad_m_basis(bc, m)
        cell2dof = self.cell_to_dof()
        val = bm.einsum('cqlg, cl->cqg', gmphi, uh[cell2dof])
        return val

    def interpolation(self, flist):
        """                                                                        
        @breif 对函数进行插值，其中 flist 是一个长度为 4m+1 的列表，flist[k] 是 f 的 k
            阶导组成的列表, 如 m = 1 时，flist =                                   
            [                                                                      
              [f],                                                                 
              [fx, fy, fx],                                                        
              [fxx, fxy, fxz, fyy, fyz, fzz],                                      
              [fxxx, fxxy, fxxz, fxyy, fxyz, fxzz, fyyy, fyyz, fyzz, fzzz]         
              [fxxxx, fxxxy, fxxxz, fxxyy, fxxyz, fxxzz, fxyyy, fxyyz, fxyzz, fxzzz, fyyyy, fyyyz, fyyzz, fyzzz, fzzzz]
            ]                                                                      
        """       
        m = self.m
        p = self.p
        mesh = self.mesh
        fI = self.function()
        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        NN   = mesh.number_of_nodes()
        NE   = mesh.number_of_edges()
        NF   = mesh.number_of_faces()
        NC   = mesh.number_of_cells()

        n2id = self.node_to_dof() # (NN, nndof)
        e2id = self.edge_to_internal_dof()
        f2id = self.face_to_internal_dof()
        c2id = self.cell_to_internal_dof()
        c2d = self.cell_to_dof()
        nndof = self.number_of_internal_dofs('node')


        nframe, eframe, _ = self.get_frame()


        # node  
        fI[n2id[:, 0]] = flist[0](node)
        k = 1
        for r in range(1, 4*m+1):
            symidx, num = symmetry_index(3, r)
            val = flist[r](node)
            midx = mesh.multi_index_matrix(r, 2)
            mul = bm.repeat(midx, num.astype(int), axis=0)
            #num = num**2 # 这里为什么要平方
            kkk = bm.repeat(bm.arange(num.shape[0]), num.astype(int), axis=0)
            #print(kkk)
            kk = 0 
            for idx in mul:
                nnn = symmetry_span_array(nframe, idx).reshape(-1, 3**r)[:, symidx]
                #nnn = symmetry_span_array(bm.tile(bm.eye(3),(27,1,1)), idx).reshape(-1, 3**r)[:, symidx]
                fI[n2id[:, kkk[kk]+k]] += bm.sum(nnn*val*num, axis=1)
                #fI[n2id[:, kkk[kk]+k]] = bm.einsum('ni,ni,i->n', nnn, val, num)
                #print(kkk[kk]+k)
                #fI[n2id[:, k]] = bm.einsum('ni,ni,i->n', nnn, val, num)
                kk = kk+1
            k = k + num.shape[0]

        #fI[n2id[:, 0]] = flist[0](node)
        #k = 1
        #for r in range(1, 4*m+1):
        #    symidx, num = symmetry_index(3, r)
        #    val = flist[r](node)
        #    midx = mesh.multi_index_matrix(r, 2)
        #    num = num**2 # 这里为什么要平方
        #    for idx in midx:
        #        nnn = symmetry_span_array(nframe, idx).reshape(-1, 3**r)[:, symidx]
        #        #nnn = symmetry_span_array(bm.tile(bm.eye(3),(27,1,1)), idx).reshape(-1, 3**r)[:, symidx]
        #        fI[n2id[:, k]] = bm.sum(nnn*val*num, axis=1)
        #        #fI[n2id[:, k]] = bm.einsum('ni,ni,i->n', nnn, val, num)
        #        k += 1
        # edge
        locEdge = bm.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                           **self.ikwargs)
        dualEdge = bm.array([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]],
                            **self.ikwargs)
        S12m = self.dof_index["edge"]
        en = eframe[:, 1:]

        midx1d2num = lambda x: x[1]
        N = nndof*4
        for i in range(6):
            c2ei = c2e[:, i]
            e = locEdge[i]
            es = dualEdge[i]
            Se2m = bm.concatenate([a for b in S12m[i] for a in b])
            for alpha in self.multiIndex[Se2m]:
                alphae = alpha[e]
                alphaes = alpha[es]
                r = int(bm.sum(alphaes))

                bcs = mesh.multi_index_matrix(p-r, 1, **self.fkwargs)/(p-r) # (NQ, 2)
                b2l = self.bspace.bernstein_to_lagrange(p-r, 1) #(p-r+1, p-r+1)

                point = bm.einsum("qi, cid-> qcd", bcs, node[cell[:, e]])

                ffval = bm.array(flist[r](point)) # (NQ, NC, l) :l 是分量个数 

                if r==0:
                    bcoeff = bm.einsum("qc, iq-> ci", ffval, b2l)
                else:
                    symidx, num = symmetry_index(3, r)
                    nnn = symmetry_span_array(en[c2ei], alphaes).reshape(NC, -1)[:, symidx]
                    bcoeff = bm.einsum("cl, qcl, l, iq-> ci", nnn, ffval, num, b2l) #(NC, l)

                for ccc in range(NC):
                    Ralpha = midx1d2num(alphae)
                    fI[c2d[ccc, N]] = bcoeff[ccc, Ralpha] # 这里为什么不需要反过来
                N += 1

        # face
        locFace = bm.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                           **self.ikwargs)
        dualFace = bm.array([[0],[1],[2],[3]], **self.ikwargs)

        S2m = self.dof_index["face"]
        fn = mesh.face_unit_normal()
        c2fperm = mesh.cell_to_face_permutation(locFace=locFace) # (NC, 4, 3)

        midx2d2num = lambda a: (a[1]+a[2])*(1+a[1]+a[2])//2+a[2]

        f2id = self.face_to_internal_dof()
        for i in range(4):
            c2fi = c2f[:, i]
            f = locFace[i]
            fs = dualFace[i]
            Sfm = bm.concatenate(S2m[i])
            for alpha in self.multiIndex[Sfm]:
                alphaf = alpha[f]
                alphafs = alpha[fs]
                r = int(alphafs)

                bcs = mesh.multi_index_matrix(p-r, 2, **self.fkwargs)/(p-r) # (NQ, 3)
                b2l = self.bspace.bernstein_to_lagrange(p-r, 2) # (NQ, NQ)

                point = bm.einsum("qi, cid-> qcd", bcs, node[cell[:, f]]) # (NQ, NC, 3)
                
                ffval = bm.array(flist[r](point)) # (NQ, NC, l) :l 是分量个数
                if r==0:
                    bcoeff = bm.einsum("qc, iq-> ci", ffval, b2l)
                else:
                    symidx, num = symmetry_index(3, r)
                    nnn = symmetry_span_array(fn[c2fi, None], alphafs).reshape(-1, 3**r)[:, symidx]
                    bcoeff = bm.einsum("cl, qcl, l, iq-> ci", nnn, ffval, num,
                                       b2l) #(NC, l)

                for ccc in range(NC):
                    Ralpha = midx2d2num(alphaf)
                    fI[c2d[ccc, N]] = bcoeff[ccc, Ralpha]
                N += 1

        bcs = mesh.multi_index_matrix(p, 3, **self.fkwargs)/p
        b2l = self.bspace.bernstein_to_lagrange(p, 3)
        point = mesh.bc_to_point(bcs) #(NC, NQ, 3) 
        ffval = bm.array(flist[0](point)) # (NC, NQ, )
        bcoeff = bm.einsum("cq, iq-> ci", ffval, b2l)

        S3 = self.dof_index["cell"]
        fI[c2id] = bcoeff[:, S3]
        return fI

#    def boundary_interpolate(self, gD, uh, threshold=None):
#        mesh = self.mesh
#        m = self.m
#        p = self.p
#        isCornerNode = self.isCornerNode
#        isBdNode = mesh.boundary_node_flag()
#        isBdEdge = mesh.boundary_edge_flag()
#        isBdFace = mesh.boundary_face_flag()
#
#        node = mesh.entity('node')[isBdNode]
#        edge = mesh.entity('edge')[isBdEdge]
#        face = mesh.entity('face')[isBdFace]
#
#        NN = len(node)
#        NE = len(edge)
#        NF = len(face)
#
#        nodeframe, edgeframe, faceframe = self.get_frame()
#        nodeframe = nodeframe[isBdNode]
#        edgeframe = edgeframe[isBdEdge]
#        faceframe = faceframe[isBdFace]
#
#        n2id = self.node_to_dof()[isBdNode]
#        e2id = self.edge_to_internal_dof()[isBdEdge]
#        f2id = self.face_to_internal_dof()[isBdFace]
#        # 顶点
#        uh[n2id[:, 0]] = gD[0](node)
#        k = 1
#        for r in range(1, 4*m+1):
#            val = gD[r](node)
#            symidx, num = symmetry_index(3, r)
#            #bdnidxmap = 
#            #idx = bdnidxmap[coridx]
#            #print(idx)
#        return

        


    















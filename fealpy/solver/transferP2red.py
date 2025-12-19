from typing import Optional, Tuple, Callable, Union, TypeVar
from ..typing import TensorLike
from ..backend import bm
from ..mesh import TriangleMesh
from ..functionspace import LagrangeFESpace
from ..sparse import csr_matrix


CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]

def indofP2(mesh: TriangleMesh, threshold=None, return_index=False, tensor_mesh=False):
    space = LagrangeFESpace(mesh, p=2)
    isDDof = space.is_boundary_dof()
    points = mesh.interpolation_points(p=2)
    if tensor_mesh:
        points = bm.concat([points, bm.zeros((len(points),1), dtype=bm.float64)], axis=1)

    index_dof = bm.arange(len(points))[isDDof]
    bd_point = points[isDDof] 
    if tensor_mesh:
        flag = threshold(bd_point, dim=2)
    else:
        flag = threshold(bd_point)
    index_dof = index_dof[flag]

    bd_flag = bm.zeros((len(points),), dtype=bm.bool)
    bm.set_at(bd_flag, index_dof, True)
    if return_index:
        return ~bd_flag, index_dof
    return ~bd_flag

def transferP2red(mesh: TriangleMesh, level:int, 
                  threshold:Optional[Tuple[CoefLike,...]]=None,
                  tensor_mesh=False):
    """
    see https://github.com/jzhongmath/ifem/blob/master/transfer/transferP2red.m .

        3                       *
        * *                     8  7            
        *  *                    *     *
        5   4       ------>     * * 9 * *
        *    *                  2 *     *  * 
        *     *                 *   1   5    4
        *       *               *     * *      * 
        1* * 6 * * 2            * * 3 * * * 6 * *
    
    
    """
    # input: the coaresest grid
    if threshold == 'None':
            return mesh.uniform_refine(n=level-1, returnim=True)

    def P2red(NTc, c2i0, c2i1, Ndofc, Ndoff):
        # we just consider the middle points in fine edges. 
        elem2node2f = bm.zeros((NTc, 9), dtype=bm.int32)

        elem2node2f[:, 0] = c2i1[:NTc, 1]
        elem2node2f[:, 1] = c2i1[:NTc, 2]
        elem2node2f[:, 2] = c2i1[:NTc, 4]

        elem2node2f[:, 3] = c2i1[NTc:2*NTc, 1]
        elem2node2f[:, 4] = c2i1[NTc:2*NTc, 2]
        elem2node2f[:, 5] = c2i1[NTc:2*NTc, 4]

        elem2node2f[:, 6] = c2i1[2*NTc:3*NTc, 1]
        elem2node2f[:, 7] = c2i1[2*NTc:3*NTc, 2]
        elem2node2f[:, 8] = c2i1[2*NTc:3*NTc, 4]

        locRij = bm.array([
            [ 0,      3/8,   3/8,   0,     -1/8,   -1/8,    0,     -1/8,  - 1/8],
            [-1/8,    0,    -1/8,   3/8,    0,      3/8,   -1/8,    0,     -1/8],
            [-1/8,   -1/8,   0,    -1/8,   -1/8,    0,      3/8,    3/8,    0],
            [ 1/4,    0,     0,     3/4,    1/2,    0,      3/4,    0,      1/2],
            [ 1/2,    3/4,   0,     0,      1/4,    0,      0,      3/4,    1/2],
            [ 1/2,    0,     3/4,   0,      1/2,    3/4,    0,      0,      1/4]
        ])
        idx0 = bm.array([1, 6, 5, 2, 4, 3], dtype=bm.int32)
        idx1 = bm.array([3, 2, 1, 6, 5, 4, 9, 8, 7], dtype=bm.int32)

        locRij = locRij[idx0-1, :][:, idx1-1]
        ii = bm.zeros((40*NTc,), dtype=bm.int32)
        jj = bm.zeros((40*NTc,), dtype=bm.int32)
        ss = bm.zeros((40*NTc,), dtype=bm.float64)
        index = 0

        for i in range(6):
            for j in range(9):
                if locRij[i, j] != 0:
                    ii[index:index+NTc] = c2i0[:,i] + 1
                    jj[index:index+NTc] = elem2node2f[:,j] + 1
                    
                    # Every interior edges of the coarse grid will be computed twice.
                    # So the corresponding weight is halved.
                    if (j == 2) | (j == 4) | (j == 6):
                        ss[index:index+NTc] = locRij[i, j]
                    else:
                        ss[index:index+NTc] = locRij[i, j]/2
                    
                    index = index + NTc
        
        ii = ii[~(ii==0)] - 1
        jj = jj[~(jj==0)] - 1
        ss = ss[~(ss==0)] 
    
        # Modification for boundary middle points.
        idx = bm.concat([c2i1[:, 1], c2i1[:, 2], c2i1[:, 4]], axis=0)
        s = bm.bincount(idx, minlength=Ndoff)
        bdEdgeidxf = (s == 1)
        bdEdgeidxfinjj = bdEdgeidxf[jj]
        ss[bdEdgeidxfinjj] = 2*ss[bdEdgeidxfinjj]
    
        # The transfer operator for 1-6 nodes in the coarse grid is an identical matrix
        ii = bm.concat([bm.arange(Ndofc), ii], axis=0)
        jj = bm.concat([bm.arange(Ndofc), jj], axis=0)
    
        ss = bm.concat([bm.ones(Ndofc,), ss], axis=0)
        Pro = csr_matrix((ss, (jj, ii)), shape=(Ndoff, Ndofc))
        # space = LagrangeFESpace(mesh, p=2)
        # FreeNode = bm.nonzero(space.is_boundary_dof())[0]
        # if FreeNode is not None:
        #     FreeNonec = FreeNode[FreeNode<=Ndofc-1]
        #     Pro = Pro[FreeNode, FreeNonec]

        return Pro
    
    Pro_u = []
    for i in range(level-1):
        NTc = mesh.number_of_cells()
        c2i0 = mesh.cell_to_ipoint(p=2)
        Ndofc = mesh.number_of_global_ipoints(p=2)

        if threshold is not None:
            flag0 = indofP2(mesh, threshold, tensor_mesh=tensor_mesh)
        mesh.uniform_refine()
        if threshold is not None:
            flag1 = indofP2(mesh, threshold, tensor_mesh=tensor_mesh)
        c2i1 = mesh.cell_to_ipoint(p=2)
        Ndoff = mesh.number_of_global_ipoints(p=2)
        P = P2red(NTc, c2i0, c2i1, Ndofc, Ndoff)

        if threshold != None:
            P = P.to_scipy()[bm.to_numpy(flag1)][:,bm.to_numpy(flag0)]
        Pro_u.append(P)

    return Pro_u

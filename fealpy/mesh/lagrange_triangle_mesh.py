from typing import Union, Optional, List, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import simplex_gdof, simplex_ldof 
from .mesh_base import HomogeneousMesh, estr2dim
from .triangle_mesh import TriangleMesh


class LagrangeTriangleMesh(HomogeneousMesh):
    """
    
    Parameters:
        node(TensorLike): the coordinates of the nodes.

        cell(TensorLike): the connectivity of the cells.

        p(int, optional): the order of the Lagrange element. If p is None,
            it will be computed from cell.shape[-1].

        boundary(Boundary, optional): the boundary object of the mesh.

        surface(Surface, optional): the surface object contained the mesh.

    Attributes:

    Methods:

    Notes:

    Todos:

    """
    def __init__(self, 
                 node: TensorLike, 
                 cell: TensorLike, 
                 p: Optional[int] = None, 
                 boundary=None, 
                 surface=None):

        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        if p is None:
            NV = cell.shape[-1]
            self.p = int(-3 + bm.sqrt(1 + 8 * NV)) // 2 
        else:
            NV = (p + 1) * (p + 2) // 2
            if cell.shape[-1] != NV:
                raise ValueError(f"cell.shape[-1] != {NV}, p = {p}.")
            else:
                self.p = p

        self.node = node
        self.cell = cell

        self.surface = surface
        self.boundary = boundary

        self.construct_local_edge()
        self.localFace = self.localEdge
        self.construct_local_cell()
        self.construct() # construct the toplogy of the mesh

        # TODO: project the nodes to surface
        # 

        if self.boundary is not None:
            isBdNode = self.boundary_node_flag()
            bdNode, _ = self.boundary.project(node[isBdNode])
            node = bm.set_at(node, isBdNode, bdNode)

        self.meshtype = 'ltri'
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 0.5

    def construct_local_edge(self, p: Optional[int]=None) -> None:
        """
        Generate the local edges for Lagrange elements of order p.

        Parameters:
            p(int, optional): the order of the Lagrange element. If p is None,
                it will be set to self.p.

        Returns:
            None: the local edges will be stored in self.localEdge.
        """
        p = self.p if p is None else p

        TD = self.top_dimension()
        mi = bm.multi_index_matrix(p, TD)

        edge0 = bm.where(mi[:, 0] == 0)[0]
        edge1 = bm.where(mi[:, 1] == 0)[0]
        edge2 = bm.where(mi[:, 2] == 0)[0]

        self.localEdge = bm.stack([
            edge0, bm.flip(edge1), edge2], axis=0)  # (3, p+1)

    def construct_local_cell(self, p: Optional[int]=None) -> TensorLike:
        """
        Generate the local cells for Lagrange elements of order p.

        Parameters:
            p(int, optional): the order of the Lagrange element. If p is None,
                it will be set to self.p.

        Returns:
            None: the local cells will be stored in self.localCell.
        Notes:

        """

        self.localCell = None

    def number_of_corner_nodes(self) -> int:
        """
        Get the number of corner nodes in the mesh.

        Returns:
            int: the number of corner nodes.
        """
        return self.cell_corner_node_flag().sum()

    def cell_corner_node_flag(self) -> TensorLike:
        """
        Get the flag array of the corner nodes in each cell.

        Returns:
            TensorLike: a boolean array of shape (NN, ), where NN is the number
            of nodes in the mesh. The value is True if the node is a cell corner
            node, and False otherwise.
        """

        NN = self.number_of_nodes()
        edge = self.entity('edge')

        isCornerNode = bm.zeros(NN, dtype=bm.bool)
        isCornerNode = bm.set_at(isCornerNode, edge[:, [0, -1]], True)
        return isCornerNode

    def interpolation_points(self, p: Optional[int]=None) -> TensorLike:
        """
        Fetch all p-order interpolation points on the triangle mesh.

        Parameters:
            p(int, optional): the order of the Lagrange element. If p is None,
                it will be set to self.p.

        Returns:
            TensorLike: the coordinates of the interpolation points in the
            physical space. The shape is (NIP, GD), where NIP is the number of
            interpolation points, and GD is the geometric dimension of the
            mesh.
        """

        GD = self.geo_dimension()
        node = self.entity('node')

        if p is None:
            return self.entity('node')
        elif p < 1:
            raise ValueError(f"p must be at least 1, but got {p}.")
        elif p == self.p:
            return node

        isCornerNode = self.cell_corner_node_flag()
        if p == 1:
            return node[isCornerNode]

        ipoints = []
        ipoints.append(node[isCornerNode])  # the corner nodes 

        # edge interior interpolation points 
        w = bm.multi_index_matrix(p, 1)[1:-1]/p
        ipoints.append(self.bc_to_point(w).reshape(-1, GD))

        # cell interior interpolation_points 
        if p > 2:
            TD = self.top_dimension()
            mi = self.multi_index_matrix(p, TD)
            isInCellIPoints = bm.sum(mi > 0, axis=-1) == 3
            w = mi[isInCellIPoints, :] / p
            ipoints.append(self.bc_to_point(w).reshape(-1, GD))

        return bm.concatenate(ipoints, axis=0)

    def entity_barycenter(self, 
                          etype: Union[int, str]='cell', 
                          index: Index=_S) -> TensorLike:
        """

        Parameters:
            etype(Union[int, str]): the type of the mesh entity. It can be 'cell',
                'edge', or 'node', or their corresponding integer values 2, 1, 0.

            index(Index): the index of the mesh entities.
        """
        pass

    def uniform_refine(self, n: int = 1):
        """
        Uniform refine the Lagrange triangle mesh.

        Parameters:
            n(int): the number of uniform refinements.

        Returns:

        Notes:
            
        """
        for i in range(n):
            GD = self.geo_dimension()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            edge2cell = self.edge_to_cell()

            ikwargs = bm.context(edge) # integer kwargs
            fkwargs = bm.context(node) # float kwargs

            isCornerNode = bm.zeros(len(node), dtype=bm.bool)
            isCornerNode = bm.set_at(isCornerNode, edge[:, [0, -1]], True)

            # the list container for new nodes, edges, cells, and edge2cells
            nodes = [] 
            edges = []
            cells = [] 
            edge2cells = [] 

            # the corner nodes 
            nodes.append(node[isCornerNode])


            NN = len(nodes[0]) # the number of corner nodes
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # the index of subcells in each cell
            cell2subcell = bm.arange(4 * NC, **ikwargs).reshape(NC, 4) 

            # the barycenteric coordinates of the edges
            E = bm.array([
                [1.0, 0.0], # the start point 
                [0.5, 0.5], # the middle point of the edge
                [0.0, 1.0], # the end point
                ], **fkwargs)

            # add the middle point as the new nodes
            nodes.append(self.bc_to_point(E[1]).reshape(-1, self.GD))

            # the global index of the middle points
            start = NN 
            end = start + NE 
            mid = bm.arange(start, end, **ikwargs) 

            # compute the new nodes on the two subedges        
            w = bm.multi_index_matrix(self.p, 1)[1:-1]/self.p
            w = bm.concatenate((
                w @ E[[0, 1]], 
                w @ E[[1, 2]]), axis=0)
            nodes.append(self.bc_to_point(w).reshape(-1, self.GD))
            
            # Each edge was split into two new subedges. The left cell of the
            # edge was split into four subcells. 
            # map[i, j, 0] 
            # map[i, j, 1] 
            imap = bm.array([
                [[1, 2], [2, 1]], 
                [[2, 2], [0, 1]],
                [[0, 2], [1, 1]]], **ikwargs)

            if self.p == 1:
                e0 = bm.stack((edge[:, 0], mid), axis=1)
                e1 = bm.stack((mid, edge[:, -1]), axis=1)
            else:
                nn = self.p - 1
                start = end
                end = start + 2 * nn * NE
                e = bm.arange(start, end, **ikwargs).reshape(-1, 2 * nn)
                e0 = bm.concat((edge[:, [0]], e[:, 0*nn:1*nn],  mid[:, None]), axis=1)
                e1 = bm.concat((mid[:, None], e[:, 1*nn:2*nn], edge[:, [-1]]), axis=1)
            edges.extend([e0, e1])

            e2c0 = bm.stack((
                cell2subcell[edge2cell[:, 0], imap[edge2cell[:, 2], 0, 0]],
                cell2subcell[edge2cell[:, 1], imap[edge2cell[:, 3], 1, 0]],
                imap[edge2cell[:, 2], 0, 1],
                imap[edge2cell[:, 3], 1, 1],
                ), axis=1)

            e2c1 = bm.stack((
                cell2subcell[edge2cell[:, 0], imap[edge2cell[:, 2], 1, 0]],
                cell2subcell[edge2cell[:, 1], imap[edge2cell[:, 3], 0, 0]],
                imap[edge2cell[:, 2], 1, 1],
                imap[edge2cell[:, 3], 0, 1],
                ), axis=1)

            isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])
            e2c0 = bm.set_at(e2c0, (isBdEdge, 1), e2c0[isBdEdge, 0])
            e2c0 = bm.set_at(e2c0, (isBdEdge, 3), e2c0[isBdEdge, 2])
            e2c1 = bm.set_at(e2c1, (isBdEdge, 1), e2c1[isBdEdge, 0])
            e2c1 = bm.set_at(e2c1, (isBdEdge, 3), e2c1[isBdEdge, 2])
            edge2cells.extend([e2c0, e2c1])
            

            # the barycentric coordinates of the vertices of the triangle and
            # the middle points of the three edges
            A = bm.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0]], dtype=bm.float64)

            w = bm.multi_index_matrix(self.p, 1)[1:-1]/self.p
            w = bm.concat((
                w @ A[[4, 5]], 
                w @ A[[5, 3]], 
                w @ A[[3, 4]],
                ), axis=0)
            nodes.append(self.bc_to_point(w).reshape(-1, self.GD))

            # construct the new edges in each cell
            c2e = self.cell_to_edge() + NN
            if self.p == 1:
                e0 = c2e[:, [1, 2]]
                e1 = c2e[:, [2, 0]] 
                e2 = c2e[:, [0, 1]]
            else:
                nn = self.p - 1
                start = end
                end = start + 3 * nn * NC
                e = bm.arange(start, end, **ikwargs).reshape(-1, 3 * nn)
                e0 = bm.concat((c2e[:, [1]], e[:, 0*nn:1*nn], c2e[:, [2]]), axis=1)
                e1 = bm.concat((c2e[:, [2]], e[:, 1*nn:2*nn], c2e[:, [0]]), axis=1)
                e2 = bm.concat((c2e[:, [0]], e[:, 2*nn:3*nn], c2e[:, [1]]), axis=1) 
            edges.extend([e0, e1, e2])

            e2c0 = bm.stack((
                cell2subcell[:, 3], 
                cell2subcell[:, 0],
                bm.full(NC, 0, **ikwargs),
                bm.full(NC, 0, **ikwargs),
                ), axis=1)
            e2c1 = bm.stack((
                cell2subcell[:, 3], 
                cell2subcell[:, 1],
                bm.full(NC, 1, **ikwargs),
                bm.full(NC, 0, **ikwargs),
                ), axis=1)
            e2c2 = bm.stack((
                cell2subcell[:, 3], 
                cell2subcell[:, 2],
                bm.full(NC, 2, **ikwargs),
                bm.full(NC, 0, **ikwargs),
                ), axis=1)
            edge2cells.extend([e2c0, e2c1, e2c2])


            icell = None
            if self.p >= 3:
                TD = self.top_dimension()
                mi = bm.multi_index_matrix(self.p, TD)
                isInCellNodes = bm.sum(mi > 0, axis=-1) == 3
                w = mi[isInCellNodes, :] / self.p
                nn = len(w)
                w = bm.concat((
                    w @ A[[0, 5, 4]], 
                    w @ A[[1, 3, 5]], 
                    w @ A[[2, 4, 3]], 
                    w @ A[[3, 4, 5]],
                    ), axis=0)
                nodes.append(self.bc_to_point(w).reshape(-1, self.GD))
                start = end
                end = start + 4 * nn * NC
                icell = bm.arange(start, end, **ikwargs).reshape(-1, nn)

            node = bm.concat(nodes, axis=0)
            edge = bm.concat(edges, axis=0)
            edge2cell = bm.concat(edge2cells, axis=0)

            cell2edge = bm.zeros((4*NC, 3), **ikwargs)
            cell2edge = bm.set_at(cell2edge, (edge2cell[:, 0], edge2cell[:, 2]), range(len(edge)))
            cell2edge = bm.set_at(cell2edge, (edge2cell[:, 1], edge2cell[:, 3]), range(len(edge)))

            self.node = node
            self.face = edge
            self.edge = edge 
            self.edge2cell = edge2cell
            self.face2cell = edge2cell
            self.cell2face = cell2edge
            self.cell2edge = cell2edge

            # construct the new ells
            self.construct_global_cell(icell, **ikwargs)


    def construct_global_cell(self, icell: TensorLike, **ikwargs):
        """
        Construct the new cells for the Lagrange triangle mesh after uniform
        refinement.

        Parameters:
            icell(TensorLike):

        Notes:
            
        """
        p = self.p 
        NC = self.number_of_cells()
        ldof = (p + 1) * (p + 2) // 2
        cell = bm.zeros((4*NC, ldof), **ikwargs)

        TD = self.top_dimension()
        mi = self.multi_index_matrix(p, TD)
        idx0, = bm.nonzero(mi[:, 0] == 0)
        idx1, = bm.nonzero(mi[:, 1] == 0)
        idx2, = bm.nonzero(mi[:, 2] == 0)
        edge2cell = self.edge_to_cell()

        edge = self.entity('edge')

        flag = edge2cell[:, 2] == 0
        cell = bm.set_at(cell, (edge2cell[flag, 0][:, None], idx0), edge[flag])

        flag = edge2cell[:, 2] == 1
        idx1_ = bm.flip(idx1, axis=0)
        cell = bm.set_at(cell, (edge2cell[flag, 0][:, None], idx1_), edge[flag])

        flag = edge2cell[:, 2] == 2
        cell = bm.set_at(cell, (edge2cell[flag, 0][:, None], idx2), edge[flag])

        iflag = edge2cell[:, 0] != edge2cell[:, 1]
        flag = iflag & (edge2cell[:, 3] == 0)
        idx0_ = bm.flip(idx0, axis=0)
        cell = bm.set_at(cell, (edge2cell[flag, 1][:, None], idx0_), edge[flag])

        flag = iflag & (edge2cell[:, 3] == 1)
        cell = bm.set_at(cell, (edge2cell[flag, 1][:, None], idx1), edge[flag])

        flag = iflag & (edge2cell[:, 3] == 2)
        idx2_ = bm.flip(idx2, axis=0)
        cell = bm.set_at(cell, (edge2cell[flag, 1][:, None], idx2_),  edge[flag])

        if self.p >= 3:
            flag = bm.sum(mi > 0, axis=1) == 3
            cell = bm.set_at(cell, (..., flag), icell)

        self.cell = cell


    def quadrature_formula(self, q: int, etype: Union[int, str]='cell'):
        """

        Paramerters:
            q(int): the index(>=1) of the quadrature formula.

        Return:
            A quadrature formula object for the specified entity type.

        Notes:
            
        """

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 2:
            from ..quadrature import TriangleQuadrature
            quad = TriangleQuadrature(q)
        elif etype == 1:
            from ..quadrature import GaussLegendreQuadrature
            quad = GaussLegendreQuadrature(q)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    def bc_to_point(self, bc: TensorLike, index: Index=_S):
        """

        Parameters:
            bc(TensorLike): the barycentric coordinates of the integration points.
            index(Index): the index of the mesh entities.

        Returns:
            TensorLike: the coordinates of the integration points in the
            physical space.
        """
        node = self.node
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index=index) # 
        phi = self.shape_function(bc) # (NC, NQ, NVC)
        p = bm.einsum('c...n, cni -> c...i', phi, node[entity])
        return p
    
    def shape_function(self, bc: TensorLike, p: int=None, variables='x',index: Index=_S):
        """
        Parameters:
            bc(TensorLike): the barycentric coordinates of the integration points.
            p(int): the order of the Lagrange polynomial.
            variables(str): 'u' for reference variables (xi, eta), 'x' for physical variables (x, y).
            index(Index): the index of the mesh entities.

        Returns:
            TensorLike: the shape function values at the integration points.

        Notes:
            
        Todos:
            
        """
        p = self.p if p is None else p 
        phi = bm.simplex_shape_function(bc, p=p)
        if variables == 'u':
            return phi
        elif variables == 'x':
            return phi[None, ...]

    def grad_shape_function(self, bc: TensorLike, p: int=None, 
                            index: Index=_S, variables='x'):
        """

        Parameters:
            bc(TensorLike): the barycentric coordinates of the integration
            points.

            p(int): the order of the Lagrange polynomial.

            variables(str): 'u' for reference variables (xi, eta), 'x' for 
            physical variables (x, y).

            index(Index): the index of the mesh entities.

        Returns:
            TensorLike: the gradient of the shape function at the integration
            points.

        Notes:
            lambda_0 = 1 - xi - eta
            lambda_1 = xi
            lambda_2 = eta

        Todos:


        """
        p = self.p if p is None else p 
        TD = bc.shape[-1] - 1
        if TD == 2:
            Dlambda = bm.array([[-1, -1], [1, 0], [0, 1]], dtype=bm.float64)
        else:
            Dlambda = bm.array([[-1], [1]], dtype=bm.float64)
        R = bm.simplex_grad_shape_function(bc, p=p) # (NQ, ldof, TD+1)
        gphi = bm.einsum('qij, jn -> qin', R, Dlambda) # (NQ, ldof, TD)
        
        if variables == 'u':
            return gphi[None, :, :, :] #(1, ..., ldof, TD)
        elif variables == 'x':
            J = self.jacobi_matrix(bc, index=index)
            G = self.first_fundamental_form(J)
            d = bm.linalg.inv(G)
            gphi = bm.einsum('cqkm, cqmn, qln -> cqlk', J, d, gphi) 
            return gphi

    def number_of_local_ipoints(self, p:int, iptype:Union[int, str]='cell'):
        """
        """
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p:int):
        """
        """
        NN = self.number_of_corner_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        num = (NN, NE, NC)
        return simplex_gdof(p, num) 

    def cell_to_ipoint(self, p:int, index:Index=_S) -> TensorLike:
        """
        Construct the map matrix from cells to interpolation points.

        Parameters:
            p(int): the order of the Lagrange polynomial.
            index(Index): the index of the mesh cells.

        Returns:
            TensorLike: the map matrix from cells to interpolation points.
            The shape is (NC, ldof), where NC is the number of cells and ldof is
            the number of local interpolation points.
        """
        cell = self.entity('cell')
        ikwargs = bm.context(cell)

        if p == 1:
            return cell[:, [0, -self.p-1, -1]][index]

        TD = self.top_dimension()
        mi = self.multi_index_matrix(p, TD)
        idx0, = bm.nonzero(mi[:, 0] == 0)
        idx1, = bm.nonzero(mi[:, 1] == 0)
        idx2, = bm.nonzero(mi[:, 2] == 0)

        face2cell = self.face_to_cell()
        NN = self.number_of_corner_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')

        c2p = bm.zeros((NC, ldof), **ikwargs)

        flag = face2cell[:, 2] == 0
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx0), e2p[flag])

        flag = face2cell[:, 2] == 1
        idx1_ = bm.flip(idx1, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx1_), e2p[flag])

        flag = face2cell[:, 2] == 2
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx2), e2p[flag])

        iflag = face2cell[:, 0] != face2cell[:, 1]
        flag = iflag & (face2cell[:, 3] == 0)
        idx0_ = bm.flip(idx0, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx0_), e2p[flag])

        flag = iflag & (face2cell[:, 3] == 1)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx1), e2p[flag])

        flag = iflag & (face2cell[:, 3] == 2)
        idx2_ = bm.flip(idx2, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx2_),  e2p[flag])

        cdof = (p-1)*(p-2)//2
        flag = bm.sum(mi > 0, axis=1) == 3
        val = NN + NE*(p-1) + bm.arange(NC*cdof, **ikwargs).reshape(NC, cdof)
        c2p = bm.set_at(c2p, (..., flag), val)
        return c2p[index]

    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        """Get the relationship between edges and integration points."""
        NN = self.number_of_corner_nodes()
        NE = self.number_of_edges()
        edge = self.edge[index]
        ikwargs = bm.context(edges)
        indices = bm.arange(NE, **ikwargs)[index]
        return bm.concatenate([
            edges[:, [0]],
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **ikwargs) + NN,
            edges[:, -1].reshape(-1, 1),
        ], axis=-1)

    def face_to_ipoint(self, p: int, index: Index=_S):
        return self.edge_to_ipoint(p, index)
 
    def entity_measure(self, etype=2, index:Index=_S):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return bm.zeros(1, dtype=bm.float64)
        else:
            raise ValueError(f"entity type:{etype} is erong!")
        
    def cell_area(self, q=None, index: Index=_S):
        """
        Calculate the area of a cell.
        """
        p = self.p
        q = p if q is None else q
        GD = self.geo_dimension()

        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if GD == 3:
            n = bm.sqrt(bm.sum(n**2, axis=-1)) # (NC, NQ)
        a = bm.einsum('q, cq -> c', ws, n)/2.0
        return a

    def edge_length(self, q=None, index: Index=_S):
        """
        Calculate the length of the side.
        """
        p = self.p
        q = p if q is None else q
        qf = self.quadrature_formula(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        J = self.jacobi_matrix(bcs, index=index)
        l = bm.sqrt(bm.sum(J**2, axis=(-1, -2)))
        a = bm.einsum('q, cq -> c', ws, l)
        return a

    def cell_unit_normal(self, bc: TensorLike, index: Index=_S):
        """
        When calculating the surface,the direction of the unit normal at the integration point. 
        """
        J = self.jacobi_matrix(bc, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if self.GD == 3:
            l = bm.sqrt(bm.sum(n**2, axis=-1, keepdims=True))
            n /= l
        return n

    def jacobi_matrix(self, bc: TensorLike, index: Index=_S, return_grad=False):
        """
        @berif 计算参考单元 （xi, eta) 到实际 Lagrange 三角形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index)
        gphi = self.grad_shape_function(bc, variables='u')
        J = bm.einsum(
                'cin, cqim -> cqnm',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NC,NQ,ldof,TD)
        if return_grad is False:
            return J #(NC,NQ,GD,TD)
        else:
            return J, gphi

    # fundamental form
    def first_fundamental_form(self, J: TensorLike, index: Index=_S):
        """
        Compute the first fundamental form of a mesh surface at integration points.
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = bm.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = bm.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        return G
 
    # tools
    def integral(self, f, q=3, celltype=False) -> TensorLike:
        """
        @brief 在网格中数值积分一个函数
        """
        GD = self.geo_dimension()
        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)
        
        rm = self.reference_cell_measure()
        J = self.jacobi_matrix(bcs)
        G = self.first_fundamental_form(J) 
        d = bm.sqrt(bm.linalg.det(G)) # 第一基本形式开方

        if callable(f):
            if getattr(f, 'coordtype', None) == 'barycentric':
                f = f(bcs)
            else:
                f = f(ps)

        cm = self.entity_measure('cell')

        if isinstance(f, (int, float)): #  u 为标量常函数
            e = f*cm
        elif bm.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = bm.einsum('q, cq..., cq -> c...', ws*rm, f, d)
        else:
            raise ValueError(f"Unsupported type of return value: {f.__class__.__name__}.")

        if celltype:
            return e
        else:
            return bm.sum(e)

    def error(self, u, v, q=3, power=2, celltype=False) -> TensorLike:
        """
        @brief Calculate the error between two functions.
        """
        GD = self.geo_dimension()
        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        rm = self.reference_cell_measure()
        J = self.jacobi_matrix(bcs)
        G = self.first_fundamental_form(J) 
        d = bm.sqrt(bm.linalg.det(G)) # 第一基本形式开方

        if callable(u):
            if getattr(u, 'coordtype', None) == 'barycentric':
                u = u(bcs)
            else:
                u = u(ps)

        if callable(v):
            if getattr(v, 'coordtype', None) == 'barycentric':
                v = v(bcs)
            else:
                v = v(ps)
        cm = self.entity_measure('cell')
        NC = self.number_of_cells()
        if v.shape[-1] == NC:
            v = bm.swapaxes(v, 0, -1)
        #f = bm.power(bm.abs(u - v), power)
        f = bm.abs(u - v)**power
        if len(f.shape) == 1:
            f = f[:, None]

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif bm.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = bm.einsum('q, cq..., cq -> c...', ws*rm, f, d)

        if celltype is False:
            #e = bm.power(bm.sum(e), 1/power)
            e = bm.sum(e)**(1/power)
        else:
            e = bm.power(bm.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )  
    
    # 可视化
    def vtk_cell_type(self, etype='cell'):
        """
        @berif  返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index: Index=_S, fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=bm.float64)), axis=1)

        cell = self.entity(etype, index)
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]), axis=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
    
    @classmethod
    def from_box_with_circular_holes(cls, 
                                     box: TensorLike = [0, 1, 0, 1], 
                                     holes: TensorLike =[[0.0, 0.0, 0.5]], 
                                     p: int = 1, 
                                     h: float =0.1,
                                     ftype = bm.float64, 
                                     itype = bm.int32): 
        """
        Create a Lagrange triangle mesh from a rectangular box with circular
        holes.

        Parameters:
            box (list | TensorLike): The coordinates of the box in the format [xmin, xmax, ymin, ymax].
            holes (list | TensorLike): A list of tuples representing the circular holes in the format [(cx, cy, r), ...].
            p (int): The polynomial order of the Lagrange elements.
            h (float): The target mesh size for the mesh generation.
        """
        import gmsh

        assert len(box) == 4, "box must be [xmin, xmax, ymin, ymax]"
        xmin, xmax, ymin, ymax = map(float, box)
        assert p in (1, 2), "Only p=1 or p=2 supported in this minimal version."
        assert h > 0, "h must be positive."

        print("[step] init gmsh")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        try:
            print("[step] build geometry")
            gmsh.model.add("box_with_holes")
            xmin, xmax, ymin, ymax = map(float, box)

            # 外矩形 (4 条线 + loop)
            p1 = gmsh.model.occ.addPoint(xmin, ymin, 0)
            p2 = gmsh.model.occ.addPoint(xmax, ymin, 0)
            p3 = gmsh.model.occ.addPoint(xmax, ymax, 0)
            p4 = gmsh.model.occ.addPoint(xmin, ymax, 0)
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)
            outer_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

            # 每个圆洞 (圆弧 loop)
            hole_loops = []
            for cx, cy, r in holes or []:
                circle = gmsh.model.occ.addCircle(cx, cy, 0, r)
                loop = gmsh.model.occ.addCurveLoop([circle])
                hole_loops.append(loop)

            # 平面 (带孔)
            surf = gmsh.model.occ.addPlaneSurface([outer_loop] + hole_loops)

            gmsh.model.occ.synchronize()

            print("[step] mesh options & generate")
            gmsh.option.setNumber("Mesh.ElementOrder", p)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
            gmsh.model.mesh.generate(2)

            # --- minimal stats ---
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node = bm.from_numpy(node_coords.reshape(-1, 3)[:, 0:2]) # only 2D

            tri_type = gmsh.model.mesh.getElementType("triangle", p)
            types, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
            print(f"[step] found {len(elemTags)} elements of type {tri_type} with {len(node_tags)} nodes")
            cell = None
            for etype, cell in zip(types, elemNodeTags):
                if etype == tri_type:
                    nn = gmsh.model.mesh.getElementProperties(etype)[3]
                    cell = bm.array(cell, dtype=itype).reshape(-1, nn) - 1
                    break

        finally:
            print("[step] finalize gmsh")
            gmsh.finalize()
            NN = len(node)
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode = bm.set_at(isValidNode, cell, True)
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap = bm.set_at(idxMap, isValidNode, bm.arange(isValidNode.sum(), dtype=bm.int64))
            cell = idxMap[cell]

            if p == 2:
                cell = cell[:, [0, 3, 5, 1, 4, 2]] # reorder for p=2

            return cls(node, cell, p=p)


    @classmethod
    def from_box(cls, box, p: int, nx=2, ny=2):
        mesh = TriangleMesh.from_box(box, nx, ny)
        return cls.from_triangle_mesh(mesh, p)

    @classmethod
    def from_curve_triangle_mesh(cls, mesh, p: int, curve=None):
        """
        """
        init_node = mesh.entity('node')

        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if curve is not None:
            boundary_edge = mesh.boundary_edge_flag()
            e2p = mesh.edge_to_ipoint(p)[boundary_edge].flatten()

            init_node[:], _ = curve.project(init_node) 
            node[e2p], _ = curve.project(node[e2p])

        lmesh = cls(node, cell, p=p, construct=True)
        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell2edge
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh

    @classmethod
    def from_triangle_mesh(cls, mesh, p: int, surface=None, boundary=None):
        """
        """

        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)

        mesh = cls(node, cell, p=p, boundary=boundary, surface=surface)

        return mesh


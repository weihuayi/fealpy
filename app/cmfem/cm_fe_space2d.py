from math import comb, factorial

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor

from cm_fe_space import CmFESpace
from symmetric_tensor import SymmetricTensor, symmetric_span_array, symmetry_multiplicity


class CmDof2d:
    """Number 2D DoFs in the local order given by ``lattice.permutation``.

    Global DoFs are ordered by vertices, edge interiors, and cell interiors.
    Edge directions follow the endpoint order in ``mesh.edge``.
    """

    def __init__(self, mesh, lattice):
        self.mesh = mesh
        self.lattice = lattice
        self._build_global_numbering()

    def _build_global_numbering(self):
        lattice = self.lattice

        mesh = self.mesh
        c2e  = mesh.cell_to_edge()
        cell = mesh.entity('cell')
        edge = mesh.entity('edge')

        nn   = mesh.number_of_nodes()
        ne   = mesh.number_of_edges()
        nc   = mesh.number_of_cells()

        nv = self.number_of_internal_dofs('node')
        nd = self.number_of_internal_dofs('edge')
        ni = self.number_of_internal_dofs('cell')

        N = 0
        self._n2dof  = bm.arange(N, N+nn*nv, dtype=bm.int32).reshape(nn, nv)
        N += nn*nv
        self._e2dof  = bm.arange(N, N+ne*nd, dtype=bm.int32).reshape(ne, nd)
        N += ne*nd
        self._c2idof = bm.arange(N, N+nc*ni, dtype=bm.int32).reshape(nc, ni)
        self._gdof = N + nc*ni

        blocks = [self._n2dof[cell].reshape(nc, 3*nv)]
        local_edge = lattice.simplex.subsimplices[1]
        for e, f in enumerate(local_edge):
            reverse = cell[:, f[0]] != edge[c2e[:, e], 0]
            start = 0
            for layer in lattice.layers[f]:
                size = len(layer)
                ids = self._e2dof[c2e[:, e], start:start+size]
                blocks.append(bm.where(reverse[:, None], bm.flip(ids, axis=1), ids))
                start += size
        blocks.append(self._c2idof)
        self._cell_to_dof = bm.concatenate(blocks, axis=1)

    def number_of_local_dofs(self):
        return self.lattice.multi_index.shape[0]

    def number_of_internal_dofs(self, etype):
        dim = {'node': 0, 'edge': 1, 'cell': 2}[etype]
        f = self.lattice.simplex.subsimplices[dim][0]
        return len(self.lattice.indices(f))

    def number_of_global_dofs(self):
        return self._gdof

    def node_to_dof(self):
        return self._n2dof

    def edge_to_internal_dof(self):
        return self._e2dof

    def cell_to_internal_dof(self):
        return self._c2idof

    def cell_to_dof(self):
        return self._cell_to_dof


class CmFESpace2d(CmFESpace):
    """Smooth finite element space on triangles for ``r=(r0,r1,0)``.

    The local order is ``lattice.permutation`` and agrees with the columns
    of ``cell_to_dof``. ``D`` and ``T`` store the local DoF and frame
    transformation matrices as CSRTensors with a shared sparsity pattern.
    """

    def __init__(self, mesh, p, r):
        super().__init__(mesh, p, r, CmDof2d)
        self._build_boundary_entity_flags()
        self.local_normal = self.local_normal_frame()
        self.global_normal = self.global_normal_frame()
        self.vertex_frame = self.global_normal['node']
        self.D = self.local_dof_matrix()
        self.T = self.frame_transform_matrix()

    def _build_boundary_entity_flags(self):
        """Classify interior nodes, straight-boundary nodes, and corners."""
        mesh = self.mesh
        edge = bm.tolist(mesh.entity('edge'))
        point = bm.tolist(mesh.entity('node'))
        boundary_edge = mesh.boundary_edge_flag()
        boundary_node = mesh.boundary_node_flag()
        incident = [[] for _ in point]
        for e in bm.tolist(bm.where(boundary_edge)[0]):
            a, b = edge[e]
            incident[a].append(e)
            incident[b].append(e)

        regular = [False]*len(point)
        for v, edges in enumerate(incident):
            if len(edges) != 2:
                continue
            directions = []
            for e in edges:
                a, b = edge[e]
                other = b if a == v else a
                dx = point[other][0] - point[v][0]
                dy = point[other][1] - point[v][1]
                length = (dx*dx + dy*dy)**0.5
                directions.append((dx/length, dy/length))
            regular[v] = abs(directions[0][0]*directions[1][1]
                             - directions[0][1]*directions[1][0]) < 1.0e-12

        self.boundary_edge_flag = boundary_edge
        self.boundary_node_flag = boundary_node
        self.regular_boundary_node_flag = bm.array(regular, dtype=bm.bool)
        self.corner_node_flag = boundary_node & ~self.regular_boundary_node_flag
        self.interior_node_flag = ~boundary_node

    def local_normal_frame(self):
        """Construct the local dual normal frame from the paper.

        ``normal[f]`` has shape ``(NC, 2-dim(f), 2)``. Vectors are stored
        by rows in complement order and satisfy
        ``n_i dot grad(lambda_j) = delta_ij``.
        """
        mesh = self.mesh
        points = mesh.entity('node')[mesh.entity('cell')]
        glambda = mesh.grad_lambda()
        normal = {}
        for f in self.lattice.simplex.subsimplices[0]:
            opposite = [j for j in range(3) if j not in f]
            normal[f] = points[:, opposite, :] - points[:, f[0], None, :]
        for f in self.lattice.simplex.subsimplices[1]:
            j = next(j for j in range(3) if j not in f)
            gradient = glambda[:, j, :]
            squared_norm = bm.sum(gradient**2, axis=1)
            normal[f] = (gradient / squared_norm[:, None])[:, None, :]
        return normal

    def global_normal_frame(self):
        """Construct row-wise global orthonormal frames.

        Interior nodes and corners use the Cartesian frame. Straight-boundary
        nodes use ``(t,n_out)``. Interior edges use a deterministic unit
        normal and boundary edges use the outward unit normal.
        """
        mesh = self.mesh
        identity = bm.eye(2, dtype=self.ftype, device=self.device)
        vertex_frame = bm.tile(identity[None, :, :], (mesh.number_of_nodes(), 1, 1))

        edge = mesh.entity('edge')
        points = mesh.entity('node')
        tangent = points[edge[:, 1]] - points[edge[:, 0]]
        # One-column tangential QR with positive diagonal in R.
        tangent = tangent / bm.sqrt(bm.sum(tangent**2, axis=1))[:, None]
        projector = identity[None, :, :] - tangent[:, :, None]*tangent[:, None, :]
        pivot = bm.argmax(bm.sum(projector**2, axis=1), axis=1)
        normal = projector[bm.arange(len(edge), device=self.device), :, pivot]
        normal = normal / bm.sqrt(bm.sum(normal**2, axis=1))[:, None]

        midpoint = (points[edge[:, 0]] + points[edge[:, 1]])/2
        cell_point = points[mesh.entity('cell')]
        centroid = bm.sum(cell_point, axis=1)/3
        adjacent = mesh.face_to_cell()[:, 0]
        inward = bm.sum(normal*(centroid[adjacent]-midpoint), axis=1) > 0
        flip = self.boundary_edge_flag & inward
        normal = bm.where(flip[:, None], -normal, normal)

        boundary_edge = bm.where(self.boundary_edge_flag)[0]
        ends = edge[boundary_edge].reshape(-1)
        values = bm.repeat(normal[boundary_edge], 2, axis=0)
        normal_sum = bm.zeros((mesh.number_of_nodes(), 2), dtype=self.ftype)
        normal_sum = bm.index_add(normal_sum, ends, values, axis=0)
        length = bm.sqrt(bm.sum(normal_sum**2, axis=1))
        vertex_normal = normal_sum / bm.where(length > 0, length, 1)[:, None]
        vertex_tangent = bm.stack((vertex_normal[:, 1], -vertex_normal[:, 0]), axis=1)
        index = bm.where(self.regular_boundary_node_flag)[0]
        vertex_frame = bm.set_at(vertex_frame, (index, 0), vertex_tangent[index])
        vertex_frame = bm.set_at(vertex_frame, (index, 1), vertex_normal[index])
        return {'node': vertex_frame, 'edge': normal[:, None, :]}

    def local_dof_matrix(self):
        """Assemble the normalized DoF matrix ``D`` by entity dimension."""
        lattice = self.lattice
        nc = self.mesh.number_of_cells()
        row_indices, row_values = [], []

        # Vertex blocks: derivatives along local edge directions.
        for f in lattice.simplex.subsimplices[0]:
            v = f[0]
            i, j = (i for i in range(3) if i != v)
            for s, rows in enumerate(lattice.layers[f]):
                eta = bm.multi_index_matrix(s, 2, dtype=bm.int32)
                for alpha in bm.tolist(lattice.multi_index[rows]):
                    mask = (eta[:, i] <= alpha[i]) & (eta[:, j] <= alpha[j])
                    increments = eta[mask]
                    # Normalization makes the edge-direction coefficients geometry independent.
                    weights = [(-1)**e[v] * comb(alpha[i], e[i]) * comb(alpha[j], e[j])
                               for e in bm.tolist(increments)]
                    weights = bm.array(weights, dtype=self.ftype, device=self.device)
                    values = bm.broadcast_to(weights[None, :], (nc, len(weights)))
                    base = [0, 0, 0]
                    base[v] = alpha[v]
                    beta = increments + bm.array(base, dtype=bm.int32)
                    row_indices.append(beta)
                    row_values.append(values)

        # Edge blocks: local dual-normal derivatives and edge Bernstein coefficients.
        glambda = self.mesh.grad_lambda()
        for f in lattice.simplex.subsimplices[1]:
            j = next(i for i in range(3) if i not in f)
            normal = self.local_normal[f][:, 0, :]
            direction = bm.einsum('cd,cid->ci', normal, glambda)
            direction = bm.set_at(direction, (slice(None), j), 1.0)
            for s, rows in enumerate(lattice.layers[f]):
                if len(rows) == 0:
                    continue
                eta = bm.multi_index_matrix(s, 2, dtype=bm.int32)
                weights = [factorial(s) / (factorial(a)*factorial(b)*factorial(c))
                           for a, b, c in bm.tolist(eta)]
                weights = bm.array(weights, dtype=self.ftype, device=self.device)
                values = weights[None, :] * bm.prod(
                    direction[:, None, :]**eta[None, :, :], axis=2
                )
                for alpha in bm.tolist(lattice.multi_index[rows]):
                    base = alpha.copy()
                    base[j] = 0
                    beta = eta + bm.array(base, dtype=bm.int32)
                    row_indices.append(beta)
                    row_values.append(values)

        # Cell block: Bernstein coefficients give one unit diagonal entry per row.
        values = bm.ones((nc, 1), dtype=self.ftype, device=self.device)
        for row in lattice.indices((0, 1, 2)):
            beta = lattice.multi_index[row][None, :]
            row_indices.append(beta)
            row_values.append(values)

        # Convert Bernstein multi-indices to local columns and assemble one CSR pattern.
        indptr = [0]
        columns, data = [], []
        for beta, values in zip(row_indices, row_values):
            t = beta[:, 1] + beta[:, 2]
            original = t*(t+1)//2 + beta[:, 2]
            col = lattice.inverse_permutation[original]
            order = bm.argsort(col)
            columns.append(col[order])
            data.append(values[:, order])
            indptr.append(indptr[-1] + len(col))

        ndof = self.number_of_local_dofs()
        return CSRTensor(
            bm.array(indptr, dtype=bm.int32, device=self.device),
            bm.concatenate(columns), bm.concatenate(data, axis=1), (ndof, ndof)
        )

    def frame_transform_matrix(self):
        """Build ``T`` from the paper's symmetric tensor products so ``L=T G``."""
        lattice, mesh = self.lattice, self.mesh
        nc = mesh.number_of_cells()
        cell, c2e = mesh.entity('cell'), mesh.cell_to_edge()
        counts, columns, data = [], [], []

        # Vertex blocks: contract all equal-order alpha and beta tensors together.
        for f in lattice.simplex.subsimplices[0]:
            local_normal = self.local_normal[f]
            global_normal = self.global_normal['node'][cell[:, f[0]]]
            opposite = [i for i in range(3) if i not in f]
            for s, rows in enumerate(lattice.layers[f]):
                alpha = lattice.multi_index[rows][:, opposite]
                local_tensor = SymmetricTensor(local_normal, alpha)
                global_tensor = SymmetricTensor(global_normal, alpha)

                # ``inner`` is complete; this is the additional ``s!/beta!`` factor.
                weight = bm.array([comb(s, b) for a, b in bm.tolist(alpha)], dtype=self.ftype)
                block = local_tensor.inner(global_tensor) * weight
                col = lattice.inverse_permutation[rows]
                columns.append(bm.tile(col, len(rows)))
                data.append(block.reshape(nc, -1))
                counts.append(bm.full((len(rows),), len(rows), dtype=bm.int32))

        # Edge blocks use ``(n^s):(N^s) = (n dot N)^s``.
        for e, f in enumerate(lattice.simplex.subsimplices[1]):
            normal = self.local_normal[f]
            global_normal = self.global_normal['edge'][c2e[:, e]]
            for s, rows in enumerate(lattice.layers[f]):
                local_tensor = SymmetricTensor(normal, (s,))
                global_tensor = SymmetricTensor(global_normal, (s,))
                values = local_tensor.inner(global_tensor)
                columns.append(lattice.inverse_permutation[rows])
                data.append(bm.broadcast_to(values[:, None], (nc, len(rows))))
                counts.append(bm.ones(len(rows), dtype=bm.int32))

        # The cell-interior block is the identity.
        rows = lattice.indices((0, 1, 2))
        columns.append(lattice.inverse_permutation[rows])
        data.append(bm.ones((nc, len(rows)), dtype=self.ftype))
        counts.append(bm.ones(len(rows), dtype=bm.int32))

        indptr = bm.concatenate((bm.zeros(1, dtype=bm.int32),
                                 bm.cumsum(bm.concatenate(counts), dtype=bm.int32)))
        ndof = self.number_of_local_dofs()
        return CSRTensor(indptr, bm.concatenate(columns), bm.concatenate(data, axis=1),
                         (ndof, ndof))

    def _vertex_interpolation(self, derivatives):
        """Convert Cartesian derivatives to normalized global-frame vertex DoFs."""
        values = []
        node = self.mesh.entity('node')
        for s, rows in enumerate(self.lattice.layers[(0,)]):
            derivative = derivatives[s](node)
            if s == 0:
                derivative = derivative[..., None]
            tensor, multiplicity = symmetric_span_array(self.vertex_frame, s)
            component = self.lattice.multi_index[rows, 2]
            directional = bm.einsum(
                'nig,ng,g->ni', tensor[:, component], derivative, multiplicity
            )
            values.append(factorial(self.p-s)/factorial(self.p)*directional)
        return bm.concatenate(values, axis=1)

    def _edge_interpolation(self, derivatives, index=slice(None)):
        """Evaluate normalized interior DoFs on selected edges."""
        p, lattice = self.p, self.lattice
        values = []
        edge = self.mesh.entity('edge')[index]
        points = self.mesh.entity('node')
        normal = self.global_normal['edge'][index, 0, :]
        for s, rows in enumerate(self.lattice.layers[(1, 2)]):
            degree = p-s
            multi_index = bm.multi_index_matrix(degree, 1, dtype=bm.int32)
            bcs = multi_index / degree
            point = bm.einsum('qi,eid->eqd', bcs, points[edge])
            value = derivatives[s](point)
            if s == 0:
                normal_derivative = value
            else:
                normal_derivative = sum(
                    comb(s, j)*normal[:, None, 0]**(s-j)*normal[:, None, 1]**j
                    * value[..., j] for j in range(s+1)
                )
            collocation = self.bspace.basis(bcs, p=degree)[0]
            coefficient = bm.swapaxes(
                bm.linalg.solve(collocation, bm.swapaxes(normal_derivative, 0, 1)),
                0, 1
            )
            component = lattice.multi_index[rows, 2]
            scale = factorial(degree) / factorial(p)
            values.append(scale*coefficient[:, component])
        return bm.concatenate(values, axis=1)

    def interpolate(self, derivatives):
        """Interpolate with the normalized extended DoFs from the paper.

        ``derivatives[s](x)`` returns independent Cartesian components in the
        order ``(d_x^s, d_x^(s-1)d_y, ..., d_y^s)``. Order zero is scalar.
        """
        uh = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)
        uh = bm.set_at(uh, self.dof.node_to_dof(),
                       self._vertex_interpolation(derivatives))
        uh = bm.set_at(uh, self.dof.edge_to_internal_dof(),
                       self._edge_interpolation(derivatives))
        value = self._cell_interpolation(derivatives)
        uh = bm.set_at(uh, self.dof.cell_to_internal_dof(), value)
        return uh

    def is_boundary_dof(self):
        """Mark essential polyharmonic boundary DoFs for ``r=(2m,m,0)``."""
        m = self.r[1]
        if self.r[0] != 2*m:
            raise NotImplementedError("Boundary DoFs currently require r[0] == 2*r[1].")

        flag = bm.zeros(self.number_of_global_dofs(), dtype=bm.bool)
        edge_dof = self.dof.edge_to_internal_dof()[self.boundary_edge_flag]
        flag = bm.set_at(flag, edge_dof, True)

        node_dof = self.dof.node_to_dof()
        flag = bm.set_at(flag, node_dof[self.corner_node_flag], True)
        regular_dof = node_dof[self.regular_boundary_node_flag]
        start = 0
        for rows in self.lattice.layers[(0,)]:
            normal_order = self.lattice.multi_index[rows, 2]
            selected = bm.where(normal_order <= m)[0]
            flag = bm.set_at(flag, regular_dof[:, start+selected], True)
            start += len(rows)
        return flag

    def boundary_interpolate(self, derivatives, uh=None):
        """Interpolate the boundary lift and return it with the boundary mask."""
        boundary = self.is_boundary_dof()
        if uh is None:
            uh = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)

        node_dof = self.dof.node_to_dof()
        node_boundary = boundary[node_dof]
        vertex_value = self._vertex_interpolation(derivatives)
        uh = bm.set_at(uh, node_dof[node_boundary], vertex_value[node_boundary])

        edge_index = bm.where(self.boundary_edge_flag)[0]
        edge_dof = self.dof.edge_to_internal_dof()[edge_index]
        edge_value = self._edge_interpolation(derivatives, edge_index)
        uh = bm.set_at(uh, edge_dof, edge_value)
        return uh, boundary

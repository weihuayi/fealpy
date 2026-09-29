from itertools import permutations
from math import factorial

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor

from cm_fe_space import CmFESpace
from symmetric_tensor import (
    SymmetricTensor, symmetric_span_array, symmetry_multiplicity
)


class CmDof3d:
    """Number tetrahedral DoFs in the order ``lattice.permutation``."""

    def __init__(self, mesh, lattice):
        self.mesh = mesh
        self.lattice = lattice
        self._build_global_numbering()

    def _reference_lookup(self, dim):
        """Map ``(layer, tangential index, normal index)`` to a block position."""
        simplex = self.lattice.simplex
        f = simplex.subsimplices[dim][0]
        opposite = tuple(i for i in simplex.vertices if i not in f)
        alpha = bm.tolist(self.lattice.multi_index)
        lookup = {}
        position = 0
        for s, rows in enumerate(self.lattice.layers[f]):
            for row in bm.tolist(rows):
                key = (s, tuple(alpha[row][i] for i in f),
                       tuple(alpha[row][i] for i in opposite))
                lookup[key] = position
                position += 1
        return lookup

    def _entity_block(self, dim, f, entity_index, entities, offset, block_size):
        """Map one local subsimplex block to its global entity block."""
        lattice = self.lattice
        cell = self.mesh.entity('cell')
        alpha = lattice.multi_index
        opposite = tuple(i for i in lattice.simplex.vertices if i not in f)
        lookup = self._reference_lookup(dim)
        rows = bm.concatenate(lattice.layers[f])
        local_vertices = cell[:, list(f)]
        global_vertices = entities[entity_index]
        if dim == 0:
            global_vertices = global_vertices[:, None]

        block = bm.zeros((cell.shape[0], block_size), dtype=bm.int32)
        normal = bm.tolist(alpha[rows][:, list(opposite)])
        for permutation in permutations(range(dim+1)):
            mask = bm.all(local_vertices[:, list(permutation)] == global_vertices,
                          axis=1)
            index = bm.where(mask)[0]
            tangent = bm.tolist(alpha[rows][:, list(f)][:, list(permutation)])
            position = bm.array(
                [lookup[(sum(n), tuple(t), tuple(n))]
                 for t, n in zip(tangent, normal)],
                dtype=bm.int32
            )
            values = offset + entity_index[index, None]*block_size + position[None, :]
            block = bm.set_at(block, (index, slice(None)), values)
        return block

    def _build_global_numbering(self):
        mesh = self.mesh
        nn = mesh.number_of_nodes()
        ne = mesh.number_of_edges()
        nf = mesh.number_of_faces()
        nc = mesh.number_of_cells()
        nv = self.number_of_internal_dofs('node')
        nd = self.number_of_internal_dofs('edge')
        nfd = self.number_of_internal_dofs('face')
        ni = self.number_of_internal_dofs('cell')

        offset = 0
        self._n2dof = bm.arange(offset, offset+nn*nv, dtype=bm.int32).reshape(nn, nv)
        offset += nn*nv
        self._e2dof = bm.arange(offset, offset+ne*nd, dtype=bm.int32).reshape(ne, nd)
        offset += ne*nd
        self._f2dof = bm.arange(offset, offset+nf*nfd, dtype=bm.int32).reshape(nf, nfd)
        offset += nf*nfd
        self._c2idof = bm.arange(offset, offset+nc*ni, dtype=bm.int32).reshape(nc, ni)
        self._gdof = offset + nc*ni

        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()
        blocks = []
        for v, f in enumerate(self.lattice.simplex.subsimplices[0]):
            blocks.append(self._entity_block(
                0, f, cell[:, v], bm.arange(nn, dtype=bm.int32), 0, nv
            ))
        edge_offset = nn*nv
        for e, f in enumerate(self.lattice.simplex.subsimplices[1]):
            blocks.append(self._entity_block(
                1, f, c2e[:, e], mesh.entity('edge'), edge_offset, nd
            ))
        face_offset = edge_offset + ne*nd
        for face, f in enumerate(self.lattice.simplex.subsimplices[2]):
            blocks.append(self._entity_block(
                2, f, c2f[:, face], mesh.entity('face'), face_offset, nfd
            ))
        blocks.append(self._c2idof)
        self._cell_to_dof = bm.concatenate(blocks, axis=1)

    def number_of_local_dofs(self):
        return self.lattice.multi_index.shape[0]

    def number_of_internal_dofs(self, etype):
        dim = {'node': 0, 'edge': 1, 'face': 2, 'cell': 3}[etype]
        f = self.lattice.simplex.subsimplices[dim][0]
        return len(self.lattice.indices(f))

    def number_of_global_dofs(self):
        return self._gdof

    def node_to_dof(self):
        return self._n2dof

    def edge_to_internal_dof(self):
        return self._e2dof

    def face_to_internal_dof(self):
        return self._f2dof

    def cell_to_internal_dof(self):
        return self._c2idof

    def cell_to_dof(self):
        return self._cell_to_dof


class CmFESpace3d(CmFESpace):
    """Smooth finite element space on tetrahedra for ``r=(r0,r1,r2,0)``."""

    def __init__(self, mesh, p, r):
        super().__init__(mesh, p, r, CmDof3d)
        self.local_normal = self.local_normal_frame()
        self._build_boundary_geometry()
        self.global_normal = self.global_normal_frame()
        self.vertex_frame = self.global_normal['node']
        self.edge_frame = self.global_normal['edge']
        self.face_frame = self.global_normal['face']
        self.D = self.local_dof_matrix()
        self.T = self.frame_transform_matrix()

    def local_normal_frame(self):
        """Construct the local normal frame dual to complementary ``grad(lambda)``."""
        glambda = self.mesh.grad_lambda()
        normal = {}
        for faces in self.lattice.simplex.subsimplices[:-1]:
            for f in faces:
                opposite = [i for i in range(4) if i not in f]
                gradient = glambda[:, opposite, :]
                gram = bm.einsum('cid,cjd->cij', gradient, gradient)
                normal[f] = bm.linalg.solve(gram, gradient)
        return normal

    def _qr_normal_frame(self, entity):
        """Construct an orthonormal normal frame by QR of ordered entity vertices."""
        point = self.mesh.entity('node')
        identity = bm.eye(3, dtype=self.ftype)
        dim = entity.shape[1]-1
        tangent = point[entity[:, 1:]] - point[entity[:, :1]]
        tangent = bm.swapaxes(tangent, 1, 2)
        q, r = bm.linalg.qr(tangent, mode='reduced')
        diagonal = r[:, bm.arange(dim), bm.arange(dim)]
        sign = bm.where(diagonal < 0, -1.0, 1.0)
        q = q*sign[:, None, :]
        projector = identity[None, :, :] - bm.einsum('ndi,nei->nde', q, q)

        residual = projector
        frame = []
        entity_index = bm.arange(len(entity), dtype=bm.int32)
        for _ in range(3-dim):
            pivot = bm.argmax(bm.sum(residual**2, axis=1), axis=1)
            vector = residual[entity_index, :, pivot]
            vector = vector / bm.sqrt(bm.sum(vector**2, axis=1))[:, None]
            frame.append(vector)
            coefficient = bm.einsum('ni,nij->nj', vector, residual)
            residual = residual-vector[:, :, None]*coefficient[:, None, :]
        return bm.stack(frame, axis=1)

    def _normal_rank_and_representatives(self, face_ids, tolerance):
        """Return normal rank and the first independent boundary-face representatives."""
        normals = self.boundary_face_normal[face_ids]
        singular = bm.linalg.svd(normals, compute_uv=False)
        rank = int(bm.sum(singular > tolerance*singular[0]))
        representatives = [face_ids[0]]
        for face_id in face_ids[1:]:
            selected = self.boundary_face_normal[representatives]
            gram = bm.einsum('id,jd->ij', selected, selected)
            right = bm.einsum(
                'id,d->i', selected, self.boundary_face_normal[face_id]
            )
            coefficient = bm.linalg.solve(gram, right)
            residual = self.boundary_face_normal[face_id]
            residual = residual-bm.einsum('i,id->d', coefficient, selected)
            if float(bm.sqrt(bm.sum(residual**2))) > tolerance:
                representatives.append(face_id)
            if len(representatives) == rank:
                break
        return rank, representatives

    def _build_boundary_geometry(self, tolerance=1.0e-10):
        """Compute outward normals and classify edges and nodes by normal rank."""
        mesh = self.mesh
        point = mesh.entity('node')
        face = mesh.entity('face')
        cell = mesh.entity('cell')
        boundary_face = mesh.boundary_face_flag()

        first = point[face[:, 1]]-point[face[:, 0]]
        second = point[face[:, 2]]-point[face[:, 0]]
        normal = bm.cross(first, second)
        normal = normal/bm.sqrt(bm.sum(normal**2, axis=1))[:, None]
        face_centroid = bm.sum(point[face], axis=1)/3
        cell_centroid = bm.sum(point[cell], axis=1)/4
        adjacent = mesh.face_to_cell()[:, 0]
        inward = bm.sum(normal*(cell_centroid[adjacent]-face_centroid), axis=1) > 0
        normal = bm.where((boundary_face & inward)[:, None], -normal, normal)
        self.boundary_face_normal = normal

        edge_faces = [[] for _ in range(mesh.number_of_edges())]
        node_faces = [[] for _ in range(mesh.number_of_nodes())]
        face_to_edge = bm.tolist(mesh.face_to_edge())
        faces = bm.tolist(face)
        for face_id in bm.tolist(bm.where(boundary_face)[0]):
            for edge_id in face_to_edge[face_id]:
                edge_faces[edge_id].append(face_id)
            for node_id in faces[face_id]:
                node_faces[node_id].append(face_id)

        regular_edge = [False]*mesh.number_of_edges()
        ridge_edge = [False]*mesh.number_of_edges()
        edge_representatives = [[] for _ in edge_faces]
        for edge_id, face_ids in enumerate(edge_faces):
            if not face_ids:
                continue
            rank, representatives = self._normal_rank_and_representatives(
                face_ids, tolerance
            )
            regular_edge[edge_id] = rank == 1
            ridge_edge[edge_id] = rank == 2
            edge_representatives[edge_id] = representatives

        regular_node = [False]*mesh.number_of_nodes()
        ridge_node = [False]*mesh.number_of_nodes()
        corner_node = [False]*mesh.number_of_nodes()
        node_representatives = [[] for _ in node_faces]
        for node_id, face_ids in enumerate(node_faces):
            if not face_ids:
                continue
            rank, representatives = self._normal_rank_and_representatives(
                face_ids, tolerance
            )
            regular_node[node_id] = rank == 1
            ridge_node[node_id] = rank == 2
            corner_node[node_id] = rank == 3
            node_representatives[node_id] = representatives

        self.boundary_face_flag = boundary_face
        self.boundary_edge_flag = mesh.boundary_edge_flag()
        self.boundary_node_flag = mesh.boundary_node_flag()
        self.regular_boundary_edge_flag = bm.array(regular_edge, dtype=bm.bool)
        self.ridge_edge_flag = bm.array(ridge_edge, dtype=bm.bool)
        self.regular_boundary_node_flag = bm.array(regular_node, dtype=bm.bool)
        self.ridge_node_flag = bm.array(ridge_node, dtype=bm.bool)
        self.corner_node_flag = bm.array(corner_node, dtype=bm.bool)
        self._edge_boundary_faces = edge_representatives
        self._node_boundary_faces = node_representatives
        self._boundary_tolerance = tolerance

    def _dual_frame(self, normal):
        gram = bm.einsum('...id,...jd->...ij', normal, normal)
        return bm.linalg.solve(gram, normal)

    def global_normal_frame(self):
        """Construct interior QR frames and boundary frames fixed by boundary planes."""
        mesh = self.mesh
        point = mesh.entity('node')
        identity = bm.eye(3, dtype=self.ftype)
        node_frame = bm.tile(identity[None, :, :],
                             (mesh.number_of_nodes(), 1, 1))
        edge_frame = self._qr_normal_frame(mesh.entity('edge'))
        face_frame = self._qr_normal_frame(mesh.entity('face'))

        boundary_face = bm.where(self.boundary_face_flag)[0]
        face_frame = bm.set_at(
            face_frame, (boundary_face, 0),
            self.boundary_face_normal[boundary_face]
        )

        edge = mesh.entity('edge')
        tangent = point[edge[:, 1]]-point[edge[:, 0]]
        tangent = tangent/bm.sqrt(bm.sum(tangent**2, axis=1))[:, None]
        for edge_id in bm.tolist(bm.where(self.regular_boundary_edge_flag)[0]):
            face_id = self._edge_boundary_faces[edge_id][0]
            normal = self.boundary_face_normal[face_id]
            conormal = bm.cross(normal, tangent[edge_id])
            edge_frame = bm.set_at(edge_frame, (edge_id, 0), conormal)
            edge_frame = bm.set_at(edge_frame, (edge_id, 1), normal)

        for edge_id in bm.tolist(bm.where(self.ridge_edge_flag)[0]):
            face_ids = self._edge_boundary_faces[edge_id]
            normals = self.boundary_face_normal[face_ids]
            edge_frame = bm.set_at(edge_frame, edge_id, self._dual_frame(normals))

        for node_id in bm.tolist(bm.where(self.regular_boundary_node_flag)[0]):
            face_id = self._node_boundary_faces[node_id][0]
            normal = self.boundary_face_normal[face_id]
            projector = identity-normal[:, None]*normal[None, :]
            pivot = bm.argmax(bm.sum(projector**2, axis=0))
            tangent0 = projector[:, pivot]
            tangent0 = tangent0/bm.sqrt(bm.sum(tangent0**2))
            tangent1 = bm.cross(normal, tangent0)
            frame = bm.stack((tangent0, tangent1, normal))
            node_frame = bm.set_at(node_frame, node_id, frame)

        for node_id in bm.tolist(bm.where(self.ridge_node_flag)[0]):
            face_ids = self._node_boundary_faces[node_id]
            normals = self.boundary_face_normal[face_ids]
            tangent0 = bm.cross(normals[0], normals[1])
            tangent0 = tangent0/bm.sqrt(bm.sum(tangent0**2))
            for component in bm.tolist(tangent0):
                if abs(component) > self._boundary_tolerance:
                    if component < 0:
                        tangent0 = -tangent0
                    break
            dual = self._dual_frame(normals)
            frame = bm.concatenate((tangent0[None, :], dual), axis=0)
            node_frame = bm.set_at(node_frame, node_id, frame)

        return {'node': node_frame, 'edge': edge_frame, 'face': face_frame}

    def local_dof_matrix(self):
        """Assemble normalized vertex, edge, face, and cell blocks of ``D``."""
        lattice = self.lattice
        glambda = self.mesh.grad_lambda()
        identity = bm.eye(4, dtype=self.ftype)
        nc = self.mesh.number_of_cells()
        row_indices, row_values = [], []

        for faces in lattice.simplex.subsimplices[:-1]:
            for f in faces:
                opposite = [i for i in range(4) if i not in f]
                direction = bm.einsum('cid,cjd->cij', self.local_normal[f], glambda)

                for s, rows in enumerate(lattice.layers[f]):
                    eta_all = bm.multi_index_matrix(s, 3, dtype=bm.int32)
                    multiplicity = symmetry_multiplicity(s, 4, dtype=self.ftype)
                    expansions = {}

                    for row in bm.tolist(rows):
                        alpha = lattice.multi_index[row]
                        gamma = alpha[opposite]
                        key = tuple(bm.tolist(gamma))
                        if key not in expansions:
                            mask = eta_all[:, opposite] <= gamma[None, :]
                            mask = bm.all(mask, axis=1)

                            eta = eta_all[mask]

                            tensor0 = SymmetricTensor(direction, gamma)
                            tensor1 = SymmetricTensor(identity, eta)

                            values = tensor0.inner(tensor1) * multiplicity[mask]
                            expansions[key] = eta, values

                        eta, values = expansions[key]
                        base = bm.set_at(bm.copy(alpha), opposite, 0)
                        row_indices.append(eta + base[None, :])
                        row_values.append(values)

        values = bm.ones((nc, 1), dtype=self.ftype)
        for row in lattice.indices((0, 1, 2, 3)):
            row_indices.append(lattice.multi_index[row][None, :])
            row_values.append(values)

        indptr = [0]
        columns, data = [], []
        for beta, values in zip(row_indices, row_values):
            t = beta[:, 1] + beta[:, 2] + beta[:, 3]
            u = beta[:, 2] + beta[:, 3]
            original = t*(t+1)*(t+2)//6 + u*(u+1)//2 + beta[:, 3]
            col = lattice.inverse_permutation[original]
            order = bm.argsort(col)
            columns.append(col[order])
            data.append(values[:, order])
            indptr.append(indptr[-1] + len(col))

        ndof = self.number_of_local_dofs()
        crow = bm.array(indptr, dtype=bm.int32)
        col = bm.concatenate(columns)
        values = bm.concatenate(data, axis=1)
        return CSRTensor(crow, col, values, (ndof, ndof))

    def frame_transform_matrix(self):
        """Build ``T`` from the paper's symmetric tensor products so ``L=T G``."""
        lattice = self.lattice
        mesh = self.mesh
        nc = mesh.number_of_cells()
        entity_maps = (
            mesh.entity('cell'),
            mesh.cell_to_edge(),
            mesh.cell_to_face()
        )
        global_frames = (
            self.global_normal['node'],
            self.global_normal['edge'],
            self.global_normal['face']
        )
        counts, columns, data = [], [], []

        for dim, faces in enumerate(lattice.simplex.subsimplices[:-1]):
            for local_index, f in enumerate(faces):
                local_frame = self.local_normal[f]
                entity = entity_maps[dim][:, local_index]
                global_frame = global_frames[dim][entity]
                global_dual = self._dual_frame(global_frame)
                normal_dimension = 3-dim

                for s, rows in enumerate(lattice.layers[f]):
                    if len(rows) == 0:
                        continue
                    gamma = bm.multi_index_matrix(
                        s, normal_dimension-1, dtype=bm.int32
                    )
                    block_size = len(gamma)
                    tangent_count = len(rows)//block_size

                    local_tensor = SymmetricTensor(local_frame, gamma)
                    global_tensor = SymmetricTensor(global_dual, gamma)
                    weight = symmetry_multiplicity(
                        s, normal_dimension, dtype=self.ftype
                    )
                    block = local_tensor.inner(global_tensor)*weight

                    col = lattice.inverse_permutation[rows]
                    col = col.reshape(tangent_count, block_size)
                    col = bm.broadcast_to(
                        col[:, None, :],
                        (tangent_count, block_size, block_size)
                    )
                    values = bm.broadcast_to(
                        block[:, None, :, :],
                        (nc, tangent_count, block_size, block_size)
                    )
                    columns.append(col.reshape(-1))
                    data.append(values.reshape(nc, -1))
                    counts.append(bm.full(
                        (len(rows),), block_size, dtype=bm.int32
                    ))

        rows = lattice.indices((0, 1, 2, 3))
        columns.append(lattice.inverse_permutation[rows])
        data.append(bm.ones((nc, len(rows)), dtype=self.ftype))
        counts.append(bm.ones(len(rows), dtype=bm.int32))

        crow = bm.concatenate((
            bm.zeros(1, dtype=bm.int32),
            bm.cumsum(bm.concatenate(counts), dtype=bm.int32)
        ))
        col = bm.concatenate(columns)
        values = bm.concatenate(data, axis=1)
        ndof = self.number_of_local_dofs()
        shape = (ndof, ndof)
        return CSRTensor(crow, col, values, shape)

    def _vertex_interpolation(self, derivatives):
        """Evaluate normalized directional derivatives in global vertex frames."""
        values = []
        node = self.mesh.entity('node')
        for s, rows in enumerate(self.lattice.layers[(0,)]):
            derivative = derivatives[s](node)
            if s == 0:
                derivative = derivative[..., None]
            tensor, multiplicity = symmetric_span_array(self.vertex_frame, s)
            derivative = bm.einsum(
                'nig,ng,g->ni', tensor, derivative, multiplicity
            )
            scale = factorial(self.p-s)/factorial(self.p)
            values.append(scale*derivative)
        return bm.concatenate(values, axis=1)

    def _edge_interpolation(self, derivatives, index=slice(None)):
        """Evaluate normalized normal-derivative DoFs on selected global edges."""
        p, lattice = self.p, self.lattice
        f = lattice.simplex.subsimplices[1][0]
        edge = self.mesh.entity('edge')[index]
        point = self.mesh.entity('node')
        frame = self.global_normal['edge'][index]
        values = []

        for s, rows in enumerate(lattice.layers[f]):
            degree = p-s
            multi_index = bm.multi_index_matrix(degree, 1, dtype=bm.int32)
            bcs = multi_index/degree
            interpolation_point = bm.einsum('qi,eid->eqd', bcs, point[edge])
            derivative = derivatives[s](interpolation_point)
            if s == 0:
                derivative = derivative[..., None]

            tensor, multiplicity = symmetric_span_array(frame, s)
            directional = bm.einsum(
                'eig,eqg,g->eqi', tensor, derivative, multiplicity
            )
            collocation = self.bspace.basis(bcs, p=degree)[0]
            nq, ne, ng = len(bcs), len(edge), tensor.shape[-2]
            right = bm.swapaxes(directional, 0, 1).reshape(nq, ne*ng)
            coefficient = bm.linalg.solve(collocation, right)
            coefficient = bm.transpose(
                coefficient.reshape(nq, ne, ng), (1, 0, 2)
            )

            alpha = lattice.multi_index[rows]
            tangent_index = alpha[:, f[1]]
            normal_index = bm.tile(
                bm.arange(ng, dtype=bm.int32), len(rows)//ng
            )
            scale = factorial(degree)/factorial(p)
            values.append(
                scale*coefficient[:, tangent_index, normal_index]
            )
        return bm.concatenate(values, axis=1)

    def _face_interpolation(self, derivatives, index=slice(None)):
        """Evaluate normalized normal-derivative DoFs on selected global faces."""
        p, lattice = self.p, self.lattice
        f = lattice.simplex.subsimplices[2][0]
        face = self.mesh.entity('face')[index]
        point = self.mesh.entity('node')
        frame = self.global_normal['face'][index]
        values = []

        for s, rows in enumerate(lattice.layers[f]):
            degree = p-s
            multi_index = bm.multi_index_matrix(degree, 2, dtype=bm.int32)
            bcs = multi_index/degree
            interpolation_point = bm.einsum('qi,fid->fqd', bcs, point[face])
            derivative = derivatives[s](interpolation_point)
            if s == 0:
                derivative = derivative[..., None]

            tensor, multiplicity = symmetric_span_array(frame, s)
            directional = bm.einsum(
                'fig,fqg,g->fqi', tensor, derivative, multiplicity
            )
            collocation = self.bspace.basis(bcs, p=degree)[0]
            nq, nf = len(bcs), len(face)
            right = bm.swapaxes(directional, 0, 1).reshape(nq, nf)
            coefficient = bm.linalg.solve(collocation, right)
            coefficient = bm.swapaxes(coefficient, 0, 1)

            tangent = lattice.multi_index[rows][:, list(f)]
            u = tangent[:, 1] + tangent[:, 2]
            tangent_index = u*(u+1)//2 + tangent[:, 2]
            scale = factorial(degree)/factorial(p)
            values.append(scale*coefficient[:, tangent_index])
        return bm.concatenate(values, axis=1)

    def interpolate(self, derivatives):
        """Interpolate with the normalized extended DoFs from the paper."""
        uh = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)
        uh = bm.set_at(
            uh, self.dof.node_to_dof(),
            self._vertex_interpolation(derivatives)
        )
        uh = bm.set_at(
            uh, self.dof.edge_to_internal_dof(),
            self._edge_interpolation(derivatives)
        )
        uh = bm.set_at(
            uh, self.dof.face_to_internal_dof(),
            self._face_interpolation(derivatives)
        )

        value = self._cell_interpolation(derivatives)
        uh = bm.set_at(uh, self.dof.cell_to_internal_dof(), value)
        return uh

    def is_boundary_dof(self):
        """Mark essential boundary DoFs for ``r=(4m,2m,m,0)``, ``m=1,2``."""
        m = self.r[2]
        if m not in (1, 2) or self.r != (4*m, 2*m, m, 0):
            raise NotImplementedError(
                "3D boundary DoFs currently require r=(4m,2m,m,0), m=1 or 2."
            )

        flag = bm.zeros(self.number_of_global_dofs(), dtype=bm.bool)

        face_dof = self.dof.face_to_internal_dof()
        flag = bm.set_at(flag, face_dof[self.boundary_face_flag], True)

        edge_dof = self.dof.edge_to_internal_dof()
        flag = bm.set_at(flag, edge_dof[self.ridge_edge_flag], True)
        regular_edge_dof = edge_dof[self.regular_boundary_edge_flag]
        edge_face = self.lattice.simplex.subsimplices[1][0]
        edge_opposite = [i for i in range(4) if i not in edge_face]
        start = 0
        for rows in self.lattice.layers[edge_face]:
            gamma = self.lattice.multi_index[rows][:, edge_opposite]
            selected = bm.where(gamma[:, 1] <= m)[0]
            flag = bm.set_at(
                flag, regular_edge_dof[:, start+selected], True
            )
            start += len(rows)

        node_dof = self.dof.node_to_dof()
        flag = bm.set_at(flag, node_dof[self.corner_node_flag], True)
        node_face = self.lattice.simplex.subsimplices[0][0]
        node_opposite = [i for i in range(4) if i not in node_face]
        regular_node_dof = node_dof[self.regular_boundary_node_flag]
        ridge_node_dof = node_dof[self.ridge_node_flag]
        start = 0
        for rows in self.lattice.layers[node_face]:
            gamma = self.lattice.multi_index[rows][:, node_opposite]
            regular = bm.where(gamma[:, 2] <= m)[0]
            ridge = bm.where((gamma[:, 1] <= m) | (gamma[:, 2] <= m))[0]
            flag = bm.set_at(
                flag, regular_node_dof[:, start+regular], True
            )
            flag = bm.set_at(
                flag, ridge_node_dof[:, start+ridge], True
            )
            start += len(rows)
        return flag

    def boundary_interpolate(self, derivatives, uh=None):
        """Interpolate the full Cartesian jet onto essential boundary DoFs."""
        boundary = self.is_boundary_dof()
        if uh is None:
            uh = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)

        node_dof = self.dof.node_to_dof()
        node_boundary = boundary[node_dof]
        node_value = self._vertex_interpolation(derivatives)
        uh = bm.set_at(uh, node_dof[node_boundary], node_value[node_boundary])

        edge_index = bm.where(self.boundary_edge_flag)[0]
        edge_dof = self.dof.edge_to_internal_dof()[edge_index]
        edge_boundary = boundary[edge_dof]
        edge_value = self._edge_interpolation(derivatives, edge_index)
        uh = bm.set_at(uh, edge_dof[edge_boundary], edge_value[edge_boundary])

        face_index = bm.where(self.boundary_face_flag)[0]
        face_dof = self.dof.face_to_internal_dof()[face_index]
        face_value = self._face_interpolation(derivatives, face_index)
        uh = bm.set_at(uh, face_dof, face_value)
        return uh, boundary

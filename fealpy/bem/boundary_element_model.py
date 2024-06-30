import typing
import numpy as np


class BoundaryElementModel:
    def __init__(self, space, boundary_integrators, domain_integrators):
        self.u = None
        self.q = None
        self.space = space
        self.dintegrators = []  # 区域积分子
        self.bintegrators = []  # 边界积分子
        self.add_boundary_integrator(boundary_integrators)
        self.add_domain_integrator(domain_integrators)
        self.assembly()


    def add_domain_integrator(self, I):
        """
        @brief 增加一个或多个区域积分对象
        """
        if isinstance(I, list):
            self.dintegrators.extend(I)
        else:
            self.dintegrators.append(I)

    def add_boundary_integrator(self, I):
        """
        @brief 增加一个或多个边界积分对象
        """
        if isinstance(I, list):
            self.bintegrators.extend(I)
        else:
            self.bintegrators.append(I)

    def assembly(self):
        space = self.space
        if space.p == 0:
            gdof = space.mesh.number_of_cells()
        else:
            gdof = space.dof.number_of_global_dofs()

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(space)

        face2dof = space.dof.cell_to_dof()
        I = np.broadcast_to(np.arange(gdof, dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)

        # 整体矩阵的初始化与组装
        self._H = np.zeros((gdof, gdof))
        np.add.at(self._H, (I, J), Hij)
        np.fill_diagonal(self._H, 0.5)
        self._G = np.zeros((gdof, gdof))
        np.add.at(self._G, (I, J), Gij)
        bd_face_measure = space.mesh.entity_measure('cell')
        # TODO: 补充高次与高维情况下，对奇异积分的处理
        if space.GD == 2:
            np.fill_diagonal(self._G, (bd_face_measure * (np.log(2 / bd_face_measure) + 1) / np.pi / 2))
        # ===================================================
        f = self.dintegrators[0].assembly_cell_vector(space)
        self._f = f

        return self._H, self._G, self._f

    def boundary_condition_apply(self, boundary_condition, boundary_type=1):
        if  boundary_type == 1:
            self.G, self.F, self.u = boundary_condition.apply(self._H, self._G, self._f)
            return self.G, self.F

    def build(self):
        if self.u is not None and self.q is None:
            self.q = np.linalg.solve(self.G, self.F)


    def __call__(self, xi: np.ndarray) -> np.ndarray:
        space = self.space
        ori_shape = xi.shape[:-1]
        xi = xi.reshape((-1, space.GD))
        if space.p == 0:
            gdof = space.mesh.number_of_cells()
        else:
            gdof = space.dof.number_of_global_dofs()

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(space, xi)

        # 整体矩阵的初始化与组装
        face2dof = space.cell_to_dof()
        I = np.broadcast_to(np.arange(len(xi), dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)
        H = np.zeros((len(xi), gdof))
        np.add.at(H, (I, J), Hij)
        G = np.zeros((len(xi), gdof))
        np.add.at(G, (I, J), Gij)
        # ===================================================
        f = self.dintegrators[0].assembly_cell_vector(space, xi)

        val = G @ self.q - H @ self.u + f
        val = val.reshape(ori_shape)
        return val

    def error_calculate(self, real_solution, q=3, power=2):
        if hasattr(self.space, 'domain_mesh'):
            mesh = self.space.domain_mesh
        else:
            raise AttributeError('space has no domain_mesh.')
        qf = mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)


        if callable(real_solution):
            if not hasattr(real_solution, 'coordtype'):
                v = real_solution(ps)
            else:
                if real_solution.coordtype == 'cartesian':
                    v = real_solution(ps)
                elif real_solution.coordtype == 'barycentric':
                    v = real_solution(bcs)

        u = self(ps)
        if u.shape[-1] == 1:
            u = u[..., 0]

        if v.shape[-1] == 1:
            v = v[..., 0]

        cm = mesh.entity_measure('cell')

        f = np.power(np.abs(u - v), power)

        e = np.einsum('q, qc..., c->c...', ws, f, cm)
        e = np.power(np.sum(e), 1 / power)

        return e

    @classmethod
    def boundary_mesh_build(cls, mesh):
        from ..mesh import IntervalMesh, TriangleMesh, QuadrangleMesh
        Boundary_Mesh = {
            "TriangleMesh": IntervalMesh,
            "QuadrangleMesh": IntervalMesh,
            "UniformMesh2d": IntervalMesh,
            "TetrahedronMesh": TriangleMesh,
            "HexahedronMesh": QuadrangleMesh,
            "UniformMesh3d": QuadrangleMesh,
        }
        if type(mesh).__name__ == "UniformMesh3d":
            bd_face = mesh.ds.boundary_face()[:, [0, 2, 3, 1]]
        else:
            bd_face = mesh.ds.boundary_face()
        node = mesh.entity('node')
        old_bd_node_idx = mesh.ds.boundary_node_index()
        new_node = node[old_bd_node_idx]
        aux_idx1 = np.zeros(len(node), dtype=np.int_)
        aux_idx2 = np.arange(len(old_bd_node_idx), dtype=np.int_)
        aux_idx1[old_bd_node_idx] = aux_idx2
        new_cell = aux_idx1[bd_face]
        bd_mesh = Boundary_Mesh[type(mesh).__name__](new_node, new_cell)

        return bd_mesh
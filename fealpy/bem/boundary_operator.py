import numpy as np
from scipy.sparse import csr_matrix
from fealpy.mesh import IntervalMesh, TriangleMesh, QuadrangleMesh
from fealpy.pde.bem_model_2d import PoissonModelConstantDirichletBC2d
from fealpy.functionspace import LagrangeFESpace
from potential_grad_potential_integrator import PotentialGradPotentialIntegrator
from grad_potential_integrator import GradPotentialIntegrator
from scalar_source_integrator import ScalarSourceIntegrator
from matplotlib import pyplot as plt


class BoundaryOperator:

    Boundary_Mesh = {
        "TriangleMesh": IntervalMesh,
        "QuadrangleMesh": IntervalMesh,
        "UniformMesh2d": IntervalMesh,
        "TetrahedronMesh": TriangleMesh,
        "HexahedronMesh": QuadrangleMesh,
        "UniformMesh23d": QuadrangleMesh,
    }

    def __init__(self, space):
        self.space = space
        self._H = None
        self._G = None
        self.dintegrators = []  # 区域积分子
        self.bintegrators = []  # 边界积分子
        # 构造边界网格与空间
        mesh = space.mesh
        node = mesh.entity('node')
        old_bd_node_idx = mesh.ds.boundary_node_index()
        bd_face = mesh.ds.boundary_face()
        new_node = node[old_bd_node_idx]
        aux_idx1 = np.zeros(len(node), dtype=np.int_)
        aux_idx2 = np.arange(len(old_bd_node_idx), dtype=np.int_)
        aux_idx1[old_bd_node_idx] = aux_idx2
        cell = aux_idx1[bd_face]
        self.bd_mesh = self.Boundary_Mesh[type(mesh).__name__](new_node, cell)
        self.bd_space = LagrangeFESpace(self.bd_mesh, p=space.p)
        self.space.bd_space = self.bd_space


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
        """
        @brief 数值积分组装

        @note space 可能是以下的情形, 程序上需要更好的设计
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间张成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis()

    def assembly_for_sspace_and_vspace_with_vector_basis(self):
        # ===================================================
        bd_space = self.bd_space

        bd_gdof = bd_space.dof.number_of_global_dofs()

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(bd_space)

        face2dof = bd_space.dof.cell_to_dof()
        I = np.broadcast_to(np.arange(bd_gdof, dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)

        # 整体矩阵的初始化与组装
        self._H = np.zeros((bd_gdof, bd_gdof))
        np.add.at(self._H, (I, J), Hij)
        np.fill_diagonal(self._H, 0.5)
        self._G = np.zeros((bd_gdof, bd_gdof))
        np.add.at(self._G, (I, J), Gij)
        # ===================================================
        cell_space = self.space
        cell_gdof = cell_space.dof.number_of_global_dofs()

        f = self.dintegrators[0].assembly_cell_vector(cell_space, bd_space)
        self._f = f

        return self._H, self._G, self._f

    def assembly_for_vspace_with_scalar_basis(self):

        raise NotImplementedError

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()


if __name__ == '__main__':
    from fealpy.bem.boundary_condition import DirichletBC
    from fealpy.bem.internal_operator import InternalOperator
    pde = PoissonModelConstantDirichletBC2d()
    box = pde.domain()
    nx = 5
    ny = 5
    # 定义网格对象
    mesh = TriangleMesh.from_box(box, nx, ny)
    p = 1
    space = LagrangeFESpace(mesh, p=p)

    bd_operator = BoundaryOperator(space)
    bd_operator.add_boundary_integrator(PotentialGradPotentialIntegrator(q=p + 1))
    bd_operator.add_domain_integrator(ScalarSourceIntegrator(f=pde.source, q=p+2))

    H, G, F = bd_operator.assembly()
    bc = DirichletBC(space=space.bd_space, gD=pde.dirichlet)
    G, F = bc.apply(H, G, F)
    xi = space.bd_space.xi
    u = pde.dirichlet(xi)
    q = np.linalg.solve(G, F)

    inter_operator = InternalOperator(space)
    inter_operator.add_boundary_integrator(PotentialGradPotentialIntegrator(q=p + 1))
    inter_operator.add_domain_integrator(ScalarSourceIntegrator(f=pde.source, q=p + 2))
    inter_operator.assembly()

    print(-1)

